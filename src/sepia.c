/* sepia.c -- SEPIA CPU reference engine for Inkling (Phase 0, task 0.4).
 *
 * Single-file C11 forward pass: embeddings, hybrid local/global attention
 * with banded content-dependent relative position bias, per-layer short
 * convolutions (sconv), dense + MoE FFN, tied-free lm_head. float32
 * throughout, no quantization, no streaming, no Metal -- correctness only.
 * The spec is docs/architecture-notes.md; every non-obvious formula below
 * cites the section that justifies it.
 *
 * Numerical policy: weights/activations are stored as float32. All
 * reductions (RMSNorm's mean-of-squares, dot products / matmul rows,
 * softmax's exp-sum, the router's logsumexp, the attention-weights@V
 * weighted sum) accumulate in double and round to float32 on write. This
 * mirrors ATen's actual CPU behavior for float32 reductions (acc_type<float,
 * CPU> is double) more faithfully than naive float32 accumulation would,
 * and costs nothing at this model's scale (every reduction here is <=512
 * elements). Linear-algebra structure (which tensor multiplies which,
 * residual placement, masking) is what's being tested against the oracle,
 * not accumulation order, so we pick the most accurate option per builder's
 * discretion rather than a fragile bit-exact replica of one specific BLAS
 * kernel's summation order.
 *
 * Memory policy: this is a short-lived, single-shot CLI process. Model
 * tensors are mmap'd and referenced in place (never copied). Load-time JSON
 * trees (config.json, ref_inkling.json, the safetensors header) are never
 * explicitly freed -- the OS reclaims them at exit, and a tree-walking
 * deallocator would add code with zero observable benefit here. Per-token,
 * per-layer forward scratch (allocated inside loops that run many times per
 * process) *is* freed, to keep peak RSS bounded and valgrind-clean.
 *
 * Ported code: none -- DESIGN.md notes banded attention and sconv have no
 * parent implementation in ds4/colibri to port from.
 */
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* ============================== utilities ============================== */

static void die(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "sepia: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory (%zu bytes)", n);
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz);
    if (!p) die("out of memory (%zu x %zu bytes)", n, sz);
    return p;
}

/* Environment override for a default path, e.g. SEPIA_WEIGHTS_PATH to point
 * at a different fixture directory without a build-system change --
 * exercised by tools/test_sepia_malformed.sh, which needs to run sepia
 * against a deliberately corrupted checkpoint copy. */
static const char *env_or(const char *env_name, const char *defval) {
    const char *v = getenv(env_name);
    return (v && v[0]) ? v : defval;
}

/* Whole-file read, NUL-terminated (used for the small JSON inputs; weight
 * tensors are mmap'd separately in the safetensors loader below). */
static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open %s: %s", path, strerror(errno));
    if (fseek(f, 0, SEEK_END) != 0) die("seek %s: %s", path, strerror(errno));
    long sz = ftell(f);
    if (sz < 0) die("ftell %s failed", path);
    if (fseek(f, 0, SEEK_SET) != 0) die("seek %s: %s", path, strerror(errno));
    char *buf = xmalloc((size_t)sz + 1);
    size_t rd = fread(buf, 1, (size_t)sz, f);
    if (rd != (size_t)sz) die("short read on %s", path);
    buf[sz] = '\0';
    fclose(f);
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

/* ================================ JSON ================================= */
/* Minimal recursive-descent JSON parser, shared by config.json,
 * ref_inkling.json, and the safetensors header (also JSON). Values own
 * their strings (copied out of the source buffer), so the source buffer
 * can be freed independently of the parsed tree. \u escapes are refused
 * (not needed by any of sepia's inputs -- tensor/config keys are ASCII). */

typedef enum { JSON_NULL, JSON_BOOL, JSON_NUM, JSON_STR, JSON_ARR, JSON_OBJ } JsonType;

typedef struct JsonValue {
    JsonType type;
    double num;
    int boolean;
    char *str;
    struct JsonValue **arr_items;
    size_t arr_count;
    char **obj_keys;
    struct JsonValue **obj_vals;
    size_t obj_count;
} JsonValue;

typedef struct {
    const char *p;
    const char *end;
} JsonCursor;

static void json_skip_ws(JsonCursor *c) {
    while (c->p < c->end && (*c->p == ' ' || *c->p == '\t' || *c->p == '\n' || *c->p == '\r'))
        c->p++;
}

static JsonValue *json_new(JsonType t) {
    JsonValue *v = xcalloc(1, sizeof(JsonValue));
    v->type = t;
    return v;
}

static char *json_parse_string_raw(JsonCursor *c) {
    if (c->p >= c->end || *c->p != '"') die("json: expected string");
    c->p++;
    size_t cap = 32, len = 0;
    char *buf = xmalloc(cap);
    while (c->p < c->end && *c->p != '"') {
        char ch = *c->p++;
        if (ch == '\\') {
            if (c->p >= c->end) die("json: unterminated escape");
            char e = *c->p++;
            switch (e) {
                case '"': ch = '"'; break;
                case '\\': ch = '\\'; break;
                case '/': ch = '/'; break;
                case 'b': ch = '\b'; break;
                case 'f': ch = '\f'; break;
                case 'n': ch = '\n'; break;
                case 'r': ch = '\r'; break;
                case 't': ch = '\t'; break;
                default: die("json: unsupported escape '\\%c'", e); return NULL;
            }
        }
        if (len + 1 >= cap) {
            cap *= 2;
            char *nb = realloc(buf, cap);
            if (!nb) die("out of memory");
            buf = nb;
        }
        buf[len++] = ch;
    }
    if (c->p >= c->end) die("json: unterminated string");
    c->p++; /* closing quote */
    buf[len] = '\0';
    return buf;
}

static JsonValue *json_parse_value(JsonCursor *c) {
    json_skip_ws(c);
    if (c->p >= c->end) die("json: unexpected end of input");
    char ch = *c->p;
    if (ch == '{') {
        c->p++;
        JsonValue *v = json_new(JSON_OBJ);
        size_t cap = 8;
        v->obj_keys = xmalloc(cap * sizeof(char *));
        v->obj_vals = xmalloc(cap * sizeof(JsonValue *));
        json_skip_ws(c);
        if (c->p < c->end && *c->p == '}') {
            c->p++;
            return v;
        }
        for (;;) {
            json_skip_ws(c);
            char *key = json_parse_string_raw(c);
            json_skip_ws(c);
            if (c->p >= c->end || *c->p != ':') die("json: expected ':'");
            c->p++;
            JsonValue *val = json_parse_value(c);
            if (v->obj_count == cap) {
                cap *= 2;
                v->obj_keys = realloc(v->obj_keys, cap * sizeof(char *));
                v->obj_vals = realloc(v->obj_vals, cap * sizeof(JsonValue *));
                if (!v->obj_keys || !v->obj_vals) die("out of memory");
            }
            v->obj_keys[v->obj_count] = key;
            v->obj_vals[v->obj_count] = val;
            v->obj_count++;
            json_skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                c->p++;
                continue;
            }
            if (c->p < c->end && *c->p == '}') {
                c->p++;
                break;
            }
            die("json: expected ',' or '}'");
        }
        return v;
    } else if (ch == '[') {
        c->p++;
        JsonValue *v = json_new(JSON_ARR);
        size_t cap = 8;
        v->arr_items = xmalloc(cap * sizeof(JsonValue *));
        json_skip_ws(c);
        if (c->p < c->end && *c->p == ']') {
            c->p++;
            return v;
        }
        for (;;) {
            JsonValue *val = json_parse_value(c);
            if (v->arr_count == cap) {
                cap *= 2;
                v->arr_items = realloc(v->arr_items, cap * sizeof(JsonValue *));
                if (!v->arr_items) die("out of memory");
            }
            v->arr_items[v->arr_count++] = val;
            json_skip_ws(c);
            if (c->p < c->end && *c->p == ',') {
                c->p++;
                continue;
            }
            if (c->p < c->end && *c->p == ']') {
                c->p++;
                break;
            }
            die("json: expected ',' or ']'");
        }
        return v;
    } else if (ch == '"') {
        JsonValue *v = json_new(JSON_STR);
        v->str = json_parse_string_raw(c);
        return v;
    } else if (ch == 't') {
        if (c->end - c->p < 4 || strncmp(c->p, "true", 4) != 0) die("json: bad literal");
        c->p += 4;
        JsonValue *v = json_new(JSON_BOOL);
        v->boolean = 1;
        return v;
    } else if (ch == 'f') {
        if (c->end - c->p < 5 || strncmp(c->p, "false", 5) != 0) die("json: bad literal");
        c->p += 5;
        JsonValue *v = json_new(JSON_BOOL);
        v->boolean = 0;
        return v;
    } else if (ch == 'n') {
        if (c->end - c->p < 4 || strncmp(c->p, "null", 4) != 0) die("json: bad literal");
        c->p += 4;
        return json_new(JSON_NULL);
    } else if (ch == '-' || (ch >= '0' && ch <= '9')) {
        char *endptr;
        double d = strtod(c->p, &endptr);
        if (endptr == c->p) die("json: bad number");
        c->p = endptr;
        JsonValue *v = json_new(JSON_NUM);
        v->num = d;
        return v;
    }
    die("json: unexpected character '%c'", ch);
    return NULL;
}

static JsonValue *json_parse(const char *buf, size_t len) {
    JsonCursor c = {buf, buf + len};
    return json_parse_value(&c);
}

static const JsonValue *json_get(const JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJ) return NULL;
    for (size_t i = 0; i < obj->obj_count; i++)
        if (strcmp(obj->obj_keys[i], key) == 0) return obj->obj_vals[i];
    return NULL;
}

static int json_is_null(const JsonValue *v) { return !v || v->type == JSON_NULL; }
static double json_num(const JsonValue *v, double defval) { return (v && v->type == JSON_NUM) ? v->num : defval; }
static const char *json_str(const JsonValue *v, const char *defval) { return (v && v->type == JSON_STR) ? v->str : defval; }

/* ============================ safetensors =============================== */
/* Hand-rolled loader: mmap the file, parse the JSON header, hand back
 * pointers straight into the mapping (tensors are referenced in place, never
 * copied). Real on-disk tensor names (architecture-notes.md sec.9) are used
 * as-is -- the toy oracle's model.safetensors was written by
 * `save_pretrained()` in that same real-checkpoint layout (sec.0, sec.7's
 * "chosen layout" note), so the same lookup names work at real scale later. */

typedef struct {
    void *base;
    size_t filesize;
    JsonValue *header;
    const char *data_base;
    size_t data_size; /* filesize - (8 + header_len): bytes actually available past data_base */
} SafeTensors;

static SafeTensors st_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) die("open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat %s: %s", path, strerror(errno));
    size_t filesize = (size_t)st.st_size;
    if (filesize < 8) die("%s: truncated safetensors file", path);
    void *base = mmap(NULL, filesize, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) die("mmap %s: %s", path, strerror(errno));
    close(fd);

    uint64_t header_len;
    memcpy(&header_len, base, 8); /* little-endian u64; host is LE (x86_64/arm64) */
    if (header_len > filesize - 8) die("%s: header length exceeds file size", path);

    char *hbuf = xmalloc((size_t)header_len + 1);
    memcpy(hbuf, (const char *)base + 8, header_len);
    hbuf[header_len] = '\0';
    JsonValue *header = json_parse(hbuf, header_len);
    free(hbuf);

    SafeTensors st_ = {0};
    st_.base = base;
    st_.filesize = filesize;
    st_.header = header;
    st_.data_base = (const char *)base + 8 + header_len;
    st_.data_size = filesize - 8 - (size_t)header_len;
    return st_;
}

/* Looks up a tensor by its exact on-disk name; dies loudly if missing, of
 * the wrong dtype, whose byte length disagrees with its declared shape (a
 * cheap but effective shape sanity check without storing full shapes), or
 * -- the bounds check a reviewer found missing -- whose data_offsets fall
 * outside the bytes the mmap actually covers. A hand-crafted header can have
 * offsets that are internally self-consistent (o1-o0 matches the declared
 * shape) but still point past EOF; without this check that silently reads
 * (or, one page later, segfaults on) whatever memory happens to follow the
 * mapping instead of failing loudly. Covered by tools/test_sepia_malformed.sh.
 * `out_numel`, if non-NULL, receives the element count for the caller's own
 * cross-checks (e.g. embedding row count vs config vocab_size). */
static const float *st_find(const SafeTensors *st, const char *name, int64_t *out_numel) {
    const JsonValue *entry = json_get(st->header, name);
    if (!entry) die("safetensors: missing tensor '%s'", name);
    const char *dtype = json_str(json_get(entry, "dtype"), "");
    if (strcmp(dtype, "F32") != 0) die("safetensors: tensor '%s' has dtype %s, expected F32", name, dtype);
    const JsonValue *shape = json_get(entry, "shape");
    if (!shape || shape->type != JSON_ARR) die("safetensors: tensor '%s' missing shape", name);
    int64_t numel = 1;
    for (size_t i = 0; i < shape->arr_count; i++) numel *= (int64_t)json_num(shape->arr_items[i], 0);
    const JsonValue *offs = json_get(entry, "data_offsets");
    if (!offs || offs->type != JSON_ARR || offs->arr_count != 2)
        die("safetensors: tensor '%s' missing data_offsets", name);
    int64_t o0 = (int64_t)json_num(offs->arr_items[0], 0);
    int64_t o1 = (int64_t)json_num(offs->arr_items[1], 0);
    if (o1 - o0 != numel * (int64_t)sizeof(float))
        die("safetensors: tensor '%s' byte length disagrees with its shape", name);
    if (o0 < 0 || o1 < o0 || (uint64_t)o1 > (uint64_t)st->data_size)
        die("safetensors: tensor '%s' data_offsets [%lld,%lld) fall outside the %zu-byte data region",
            name, (long long)o0, (long long)o1, st->data_size);
    if (out_numel) *out_numel = numel;
    return (const float *)(st->data_base + o0);
}

/* ============================== config.json ============================= */
/* Only the text_config keys the forward pass needs (architecture-notes.md
 * sec.12); vision_config/audio_config are ignored entirely -- SEPIA is
 * text-only (DESIGN.md).
 *
 * Per-layer local/global and dense/sparse assignment: architecture-notes.md
 * sec.1/sec.2 present `dense_mlp_idx` and `local_layer_ids` as the fields to
 * read, and cites them as present in the real config.json. That's true of
 * the real checkpoint's *authored* config.json, but `dense_mlp_idx` (and
 * `dense_intermediate_size`) are plain **kwargs consumed by
 * InklingTextConfig.__post_init__'s `kwargs.pop(...)` -- verified directly
 * against configuration_inkling.py: they are not declared dataclass fields,
 * so they never round-trip through `save_pretrained()`. The toy oracle's
 * committed tools/oracle/tiny/config.json (itself a save_pretrained()
 * output, per make_oracle.py) demonstrably lacks both keys; only the
 * *resolved* `mlp_layer_types` array survives. This is an architecture-notes
 * defect worth fixing there: the doc should say a loader must handle both
 * representations. This loader mirrors __post_init__'s own precedence
 * exactly -- prefer the resolved array (`layer_types` / `mlp_layer_types`)
 * when present, else derive from the primitive (`local_layer_ids` /
 * `dense_mlp_idx`), else fall back to the same defaults __post_init__ uses
 * -- so it works against either representation, real or toy. */

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int vocab_size;           /* embed/unembed row count; cross-checked against the safetensors
                                * tensor shapes at load time, not otherwise used for buffer
                                * sizing (sec.8) */
    int unpadded_vocab_size;  /* lm_head output width */

    int num_attention_heads, num_key_value_heads, head_dim;             /* global (hybrid) layers */
    int swa_num_attention_heads, swa_num_key_value_heads, swa_head_dim; /* local (hybrid_sliding) layers */

    int sliding_window_size;
    int *layer_is_sliding; /* [num_hidden_layers], resolved layer_types/local_layer_ids */
    int *layer_is_sparse;  /* [num_hidden_layers], resolved mlp_layer_types/dense_mlp_idx */

    int d_rel;
    int rel_extent; /* global layers only; local layers use sliding_window_size (sec.2) */

    int conv_kernel_size;

    int dense_intermediate_size;
    int moe_intermediate_size;

    int n_routed_experts;
    int num_experts_per_tok;
    int n_shared_experts;

    double route_scale;
    double logits_mup_width_multiplier;
    double rms_norm_eps;

    double log_scaling_alpha;
    long log_scaling_n_floor;
    int has_log_scaling_floor; /* config field may be null -> log-scaling never applies (sec.4) */
} Config;

static int json_get_int(const JsonValue *tc, const char *key) {
    const JsonValue *v = json_get(tc, key);
    if (json_is_null(v)) die("config.json: missing required text_config field '%s'", key);
    return (int)json_num(v, 0);
}

static double json_get_double(const JsonValue *tc, const char *key) {
    const JsonValue *v = json_get(tc, key);
    if (json_is_null(v)) die("config.json: missing required text_config field '%s'", key);
    return json_num(v, 0);
}

static Config config_load(const char *path) {
    size_t len;
    char *buf = read_file(path, &len);
    JsonValue *root = json_parse(buf, len);
    const JsonValue *tc = json_get(root, "text_config");
    if (!tc) die("config.json: missing text_config");

    Config cfg = {0};
    cfg.hidden_size = json_get_int(tc, "hidden_size");
    cfg.num_hidden_layers = json_get_int(tc, "num_hidden_layers");
    cfg.vocab_size = json_get_int(tc, "vocab_size");
    cfg.unpadded_vocab_size = json_get_int(tc, "unpadded_vocab_size");

    cfg.num_attention_heads = json_get_int(tc, "num_attention_heads");
    cfg.num_key_value_heads = json_get_int(tc, "num_key_value_heads");
    cfg.head_dim = json_get_int(tc, "head_dim");
    cfg.swa_num_attention_heads = json_get_int(tc, "swa_num_attention_heads");
    cfg.swa_num_key_value_heads = json_get_int(tc, "swa_num_key_value_heads");
    cfg.swa_head_dim = json_get_int(tc, "swa_head_dim");

    cfg.sliding_window_size = json_get_int(tc, "sliding_window_size");

    int n = cfg.num_hidden_layers;
    cfg.layer_is_sliding = xmalloc(sizeof(int) * (size_t)n);
    const JsonValue *lt = json_get(tc, "layer_types");
    if (lt && lt->type == JSON_ARR) {
        /* Present but the wrong length is a malformed/mismatched config,
         * not a "not present" -- fall through to a weaker fallback would
         * silently paper over that instead of catching it. */
        if (lt->arr_count != (size_t)n)
            die("config.json: layer_types has %zu entries, expected num_hidden_layers=%d", lt->arr_count, n);
        for (int i = 0; i < n; i++) cfg.layer_is_sliding[i] = strcmp(json_str(lt->arr_items[i], ""), "hybrid_sliding") == 0;
    } else {
        const JsonValue *lli = json_get(tc, "local_layer_ids");
        if (lli && lli->type == JSON_ARR) {
            for (int i = 0; i < n; i++) cfg.layer_is_sliding[i] = 0;
            for (size_t i = 0; i < lli->arr_count; i++) {
                int idx = (int)json_num(lli->arr_items[i], 0);
                if (idx < 0 || idx >= n) die("config.json: local_layer_ids entry %d out of range", idx);
                cfg.layer_is_sliding[idx] = 1;
            }
        } else {
            /* configuration_inkling.py's own default: every 6th layer
             * (1-indexed) is global, the rest local. */
            for (int i = 0; i < n; i++) cfg.layer_is_sliding[i] = ((i + 1) % 6) != 0;
        }
    }

    cfg.layer_is_sparse = xmalloc(sizeof(int) * (size_t)n);
    const JsonValue *mlt = json_get(tc, "mlp_layer_types");
    if (mlt && mlt->type == JSON_ARR) {
        if (mlt->arr_count != (size_t)n)
            die("config.json: mlp_layer_types has %zu entries, expected num_hidden_layers=%d", mlt->arr_count, n);
        for (int i = 0; i < n; i++) cfg.layer_is_sparse[i] = strcmp(json_str(mlt->arr_items[i], ""), "sparse") == 0;
    } else {
        const JsonValue *dmi = json_get(tc, "dense_mlp_idx");
        int dense_mlp_idx = json_is_null(dmi) ? 0 : (int)json_num(dmi, 0); /* kwargs.pop default is 0 */
        for (int i = 0; i < n; i++) cfg.layer_is_sparse[i] = (i >= dense_mlp_idx);
    }

    cfg.d_rel = json_get_int(tc, "d_rel");
    cfg.rel_extent = json_get_int(tc, "rel_extent");

    cfg.conv_kernel_size = json_get_int(tc, "conv_kernel_size");

    /* naming trap (architecture-notes.md sec.1): dense FFN width is
     * dense_intermediate_size if present, else intermediate_size. */
    const JsonValue *div = json_get(tc, "dense_intermediate_size");
    if (!json_is_null(div))
        cfg.dense_intermediate_size = (int)json_num(div, 0);
    else
        cfg.dense_intermediate_size = json_get_int(tc, "intermediate_size");

    /* moe_intermediate_size defaults to 3072 if the checkpoint doesn't set
     * it explicitly (sec.1) -- the real config.json doesn't. */
    const JsonValue *moei = json_get(tc, "moe_intermediate_size");
    cfg.moe_intermediate_size = json_is_null(moei) ? 3072 : (int)json_num(moei, 0);

    cfg.n_routed_experts = json_get_int(tc, "n_routed_experts");
    cfg.num_experts_per_tok = json_get_int(tc, "num_experts_per_tok");
    cfg.n_shared_experts = json_get_int(tc, "n_shared_experts");

    cfg.route_scale = json_get_double(tc, "route_scale");
    cfg.logits_mup_width_multiplier = json_get_double(tc, "logits_mup_width_multiplier");
    cfg.rms_norm_eps = json_get_double(tc, "rms_norm_eps");

    cfg.log_scaling_alpha = json_get_double(tc, "log_scaling_alpha");
    const JsonValue *lsf = json_get(tc, "log_scaling_n_floor");
    if (!json_is_null(lsf)) {
        cfg.has_log_scaling_floor = 1;
        cfg.log_scaling_n_floor = (long)json_num(lsf, 0);
    }

    free(buf);
    return cfg;
}

#include "quants.h"
#include "tokenizer.h"

/* QTensor: a weight matrix stored quantized on disk (Task 14 loads these
 * in place of the float32 tensors above). Interface only here, above the
 * model structs; bodies (needing dotf) live in math primitives below. */
typedef struct { int ggml_type; const void *data; int64_t out_dim, in_dim; } QTensor;
/* y[out_dim] = W x. Per output row: dequant the row into a scratch (caller-
 * provided, in_dim floats), then double-accumulated dot -- same numerical
 * policy as linear() at src/sepia.c:694. */
static void qlinear(const QTensor *w, const float *x, float *y, float *row_scratch);
/* Dequant row `r` of W into dst (used for embedding lookups). */
static void qrow(const QTensor *w, int64_t r, float *dst);

/* RealExperts: opaque here (defined in the real-model section, Task 14) --
 * only a forward-declared pointer is needed above mlp_moe_forward, so the
 * pluggable-expert seam can be declared before its real implementation
 * exists. `real_expert_ffn` computes one selected expert's full FFN
 * (silu(gate)*up, then down_proj) from streamed quantized weights, writing
 * expert_out[hidden] -- the routing weight is applied by the caller
 * (mlp_moe_forward), identically for both the in-memory and real paths. */
struct RealExperts;
static void real_expert_ffn(const struct RealExperts *re, int expert_idx, const float *x, float *expert_out);

/* ================================ model ================================= */
/* Per-layer weight pointers plus the resolved per-layer-type dims (sec.2's
 * local/global split). Dense layers leave the moe_* pointers NULL and vice
 * versa -- a dense layer structurally has no router/experts at all (sec.1),
 * not a zero-width one. */

typedef struct {
    int is_sliding; /* hybrid_sliding vs hybrid (sec.2) */
    int is_sparse;  /* MoE vs dense MLP (sec.1) */
    int num_heads, num_kv_heads, head_dim, rel_extent;
    int q_dim, kv_dim, r_dim; /* num_heads*head_dim, num_kv_heads*head_dim, num_heads*d_rel */

    const float *attn_norm; /* [hidden] input_layernorm */
    const float *mlp_norm;  /* [hidden] post_attention_layernorm */

    const float *wq;         /* [q_dim, hidden] */
    const float *wk;         /* [kv_dim, hidden] */
    const float *wv;         /* [kv_dim, hidden] */
    const float *wr;         /* [r_dim, hidden] */
    const float *wo;         /* [hidden, q_dim] */
    const float *q_norm;     /* [head_dim] */
    const float *k_norm;     /* [head_dim] */
    const float *rel_proj;   /* [d_rel, rel_extent], row-major */
    const float *k_sconv_w;  /* [kv_dim, K] (shape [kv_dim,1,K] reinterpreted) */
    const float *v_sconv_w;  /* [kv_dim, K] */
    const float *attn_sconv_w; /* [hidden, K] */
    const float *mlp_sconv_w;  /* [hidden, K] */

    /* dense MLP (layers < dense_mlp_idx) */
    const float *dense_w13;          /* [2*dense_inter, hidden], interleaved gate/up (sec.9) */
    const float *dense_w2;           /* [hidden, dense_inter] */
    const float *dense_global_scale; /* [1] */

    /* MoE (layers >= dense_mlp_idx) */
    const float *router_w;            /* [n_routed+n_shared, hidden] */
    const float *router_bias;         /* [n_routed] (e_score_correction_bias) */
    const float *router_global_scale; /* [1] */
    const float *experts_w13;         /* [n_routed, 2*moe_inter, hidden], interleaved (sec.9) */
    const float *experts_w2;          /* [n_routed, hidden, moe_inter] */
    const float *shared_w13;          /* [n_shared, 2*moe_inter, hidden], interleaved */
    const float *shared_w2;           /* [n_shared, hidden, moe_inter] */

    /* Task 14: non-NULL in real mode only. When set, mlp_moe_forward routes
     * each SELECTED ROUTED expert's FFN through real_expert_ffn (streamed
     * quantized weights) instead of the experts_w13/experts_w2 in-memory
     * path above -- shared experts and everything else are unaffected
     * (shared_w13/shared_w2 stay resident and go through the unchanged
     * in-memory math regardless of real_exps). */
    const struct RealExperts *real_exps;
} LayerWeights;

typedef struct {
    Config cfg;
    SafeTensors st;
    LayerWeights *layers; /* [num_hidden_layers] */
    const float *embed;      /* [vocab_size, hidden] */
    const float *embed_norm; /* [hidden] */
    const float *final_norm; /* [hidden] */
    const float *unembed;    /* [vocab_size, hidden], NOT tied to embed (sec.8) */
} Model;

static const float *find_layer_tensor(const SafeTensors *st, int layer, const char *suffix) {
    char name[160];
    int n = snprintf(name, sizeof name, "model.llm.layers.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof name) die("tensor name too long for layer %d suffix %s", layer, suffix);
    return st_find(st, name, NULL);
}

static Model model_load(const char *config_path, const char *safetensors_path) {
    Model m = {0};
    m.cfg = config_load(config_path);
    m.st = st_open(safetensors_path);
    const Config *cfg = &m.cfg;

    m.layers = xcalloc((size_t)cfg->num_hidden_layers, sizeof(LayerWeights));
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        LayerWeights *lw = &m.layers[i];
        lw->is_sliding = cfg->layer_is_sliding[i];
        lw->is_sparse = cfg->layer_is_sparse[i];
        lw->num_heads = lw->is_sliding ? cfg->swa_num_attention_heads : cfg->num_attention_heads;
        lw->num_kv_heads = lw->is_sliding ? cfg->swa_num_key_value_heads : cfg->num_key_value_heads;
        lw->head_dim = lw->is_sliding ? cfg->swa_head_dim : cfg->head_dim;
        lw->rel_extent = lw->is_sliding ? cfg->sliding_window_size : cfg->rel_extent;
        lw->q_dim = lw->num_heads * lw->head_dim;
        lw->kv_dim = lw->num_kv_heads * lw->head_dim;
        lw->r_dim = lw->num_heads * cfg->d_rel;

        lw->attn_norm = find_layer_tensor(&m.st, i, "attn_norm.weight");
        lw->mlp_norm = find_layer_tensor(&m.st, i, "mlp_norm.weight");
        lw->wq = find_layer_tensor(&m.st, i, "attn.wq_du.weight");
        lw->wk = find_layer_tensor(&m.st, i, "attn.wk_dv.weight");
        lw->wv = find_layer_tensor(&m.st, i, "attn.wv_dv.weight");
        lw->wr = find_layer_tensor(&m.st, i, "attn.wr_du.weight");
        lw->wo = find_layer_tensor(&m.st, i, "attn.wo_ud.weight");
        lw->q_norm = find_layer_tensor(&m.st, i, "attn.q_norm.weight");
        lw->k_norm = find_layer_tensor(&m.st, i, "attn.k_norm.weight");
        lw->rel_proj = find_layer_tensor(&m.st, i, "attn.rel_logits_proj.proj");
        lw->k_sconv_w = find_layer_tensor(&m.st, i, "attn.k_sconv.weight");
        lw->v_sconv_w = find_layer_tensor(&m.st, i, "attn.v_sconv.weight");
        lw->attn_sconv_w = find_layer_tensor(&m.st, i, "attn_sconv.weight");
        lw->mlp_sconv_w = find_layer_tensor(&m.st, i, "mlp_sconv.weight");

        if (!lw->is_sparse) {
            lw->dense_w13 = find_layer_tensor(&m.st, i, "mlp.w13_dn.weight");
            lw->dense_w2 = find_layer_tensor(&m.st, i, "mlp.w2_md.weight");
            lw->dense_global_scale = find_layer_tensor(&m.st, i, "mlp.global_scale");
        } else {
            lw->router_w = find_layer_tensor(&m.st, i, "mlp.gate.weight");
            lw->router_bias = find_layer_tensor(&m.st, i, "mlp.gate.bias");
            lw->router_global_scale = find_layer_tensor(&m.st, i, "mlp.gate.global_scale");
            lw->experts_w13 = find_layer_tensor(&m.st, i, "mlp.experts.w13_weight");
            lw->experts_w2 = find_layer_tensor(&m.st, i, "mlp.experts.w2_weight");
            lw->shared_w13 = find_layer_tensor(&m.st, i, "mlp.shared_experts.shared_w13_weight");
            lw->shared_w2 = find_layer_tensor(&m.st, i, "mlp.shared_experts.shared_w2_weight");
        }
    }

    int64_t embed_numel = 0, unembed_numel = 0;
    m.embed = st_find(&m.st, "model.llm.embed.weight", &embed_numel);
    m.embed_norm = st_find(&m.st, "model.llm.embed_norm.weight", NULL);
    m.final_norm = st_find(&m.st, "model.llm.norm.weight", NULL);
    m.unembed = st_find(&m.st, "model.llm.unembed.weight", &unembed_numel);

    /* Cheap sanity cross-check (reviewer-requested): the embedding table's
     * row count must match config.json's vocab_size, catching a config and
     * checkpoint that were never meant to go together (wrong tensor for
     * this config, or vice versa) before it turns into silently-wrong
     * embedding lookups downstream. */
    if (embed_numel % cfg->hidden_size != 0 || embed_numel / cfg->hidden_size != cfg->vocab_size)
        die("model.llm.embed.weight has %lld elements, expected vocab_size(%d) x hidden_size(%d)",
            (long long)embed_numel, cfg->vocab_size, cfg->hidden_size);
    if (unembed_numel % cfg->hidden_size != 0 || unembed_numel / cfg->hidden_size != cfg->vocab_size)
        die("model.llm.unembed.weight has %lld elements, expected vocab_size(%d) x hidden_size(%d)",
            (long long)unembed_numel, cfg->vocab_size, cfg->hidden_size);

    return m;
}

/* =========================== math primitives ============================ */

/* x / sqrt(mean(x^2) + eps) * w, double-accumulated mean, float32 storage
 * (sec.8). Safe for out == x (in-place): all reads of x happen before any
 * write to out. */
static void rmsnorm(const float *x, const float *w, int n, float eps, float *out) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) {
        float sq = x[i] * x[i]; /* matches hidden_states.pow(2) rounding to f32 */
        acc += (double)sq;
    }
    float variance = (float)(acc / (double)n);
    float inv = 1.0f / sqrtf(variance + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * (x[i] * inv);
}

static float dotf(const float *a, const float *b, int n) {
    double acc = 0.0;
    for (int i = 0; i < n; i++) acc += (double)a[i] * (double)b[i];
    return (float)acc;
}

/* y[o] = dot(w[o,:], x) for o in [0,out_dim); w is [out_dim, in_dim] row-major. */
static void linear(const float *w, int out_dim, int in_dim, const float *x, float *y) {
    for (int o = 0; o < out_dim; o++) y[o] = dotf(w + (size_t)o * (size_t)in_dim, x, in_dim);
}

static void qrow(const QTensor *w, int64_t r, float *dst) {
    size_t row_bytes = quants_row_bytes(w->ggml_type, w->in_dim);
    const uint8_t *base = (const uint8_t *)w->data + (size_t)r * row_bytes;
    dequantize_row(w->ggml_type, base, dst, w->in_dim);
}

static void __attribute__((unused)) qlinear(const QTensor *w, const float *x, float *y, float *row_scratch) {
    for (int64_t o = 0; o < w->out_dim; o++) {
        qrow(w, o, row_scratch);
        y[o] = dotf(row_scratch, x, (int)w->in_dim);
    }
}

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
static float silu_f(float x) { return x * sigmoid_f(x); }

/* Numerically stable log(sigmoid(x)) = -softplus(-x), matching F.logsigmoid. */
static float logsigmoid_f(float x) {
    if (x >= 0.0f) return -log1pf(expf(-x));
    return x - log1pf(expf(x));
}

static int argmax_f(const float *x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > bv) {
            bv = x[i];
            best = i;
        }
    return best;
}

/* Fused gate_up_proj weights are stored on disk interleaved along the
 * output-channel axis: row 2i is gate_i, row 2i+1 is up_i
 * (architecture-notes.md sec.9's Interleave semantics, verified directly
 * against transformers/core_model_loading.py's `Interleave` op and
 * conversion_mapping.py's per-tensor dim choice). Rather than materializing
 * a deinterleaved copy, every gate/up dot product just indexes the correct
 * disk row directly -- `base` may be a per-expert-batched 3D tensor
 * ([n_experts, 2*inter, hidden], expert selects which two_inter-row block)
 * or a plain 2D one (pass expert=0, two_inter=2*inter). */
static inline const float *w13_row(const float *base, int expert, int two_inter, int hidden, int is_up, int i) {
    int c = 2 * i + (is_up ? 1 : 0);
    return base + ((size_t)expert * (size_t)two_inter + (size_t)c) * (size_t)hidden;
}

/* ============================ short convolution ========================== */
/* Depthwise causal conv1d + internal residual, one call handles both a
 * fresh chunk (hist all-zero, e.g. teacher forcing or prompt priming -- this
 * is exactly the left-zero-pad-by-K-1 causal conv, sec.5's "causal padding"
 * note) and continuing a decode session (hist populated from prior calls,
 * sec.5's "incremental decode state"). Verified against cache_utils.py's
 * `update_conv_state` that a chunk >= K-1 with empty history reduces to the
 * cache-less path bit-for-bit, so no special-casing is needed between them.
 *
 * out[t] = in[t] + sum_{k=0}^{K-1} w[k] * window[t+k], where window is the
 * concatenation of the K-1 history frames and the T new frames; hist is then
 * updated to the last K-1 frames of that concatenation. */
static void sconv_apply(const float *w /*[C,K]*/, int C, int K, float *hist /*[K-1,C], in/out*/,
                         const float *in /*[T,C]*/, int T, float *out /*[T,C]*/) {
    int Km1 = K - 1;
    float *buf = xmalloc(sizeof(float) * (size_t)(Km1 + T));
    for (int c = 0; c < C; c++) {
        for (int j = 0; j < Km1; j++) buf[j] = hist[(size_t)j * C + c];
        for (int t = 0; t < T; t++) buf[Km1 + t] = in[(size_t)t * C + c];
        for (int t = 0; t < T; t++) {
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)w[(size_t)c * K + k] * (double)buf[t + k];
            out[(size_t)t * C + c] = in[(size_t)t * C + c] + (float)acc;
        }
        for (int j = 0; j < Km1; j++) hist[(size_t)j * C + c] = buf[T + j];
    }
    free(buf);
}

/* ================================ cache ================================= */
/* Per-layer growing KV cache (post-sconv, post-norm K; post-sconv V -- what
 * future queries actually dot against, sec.2) plus the four sconv history
 * buffers (sec.5). No sliding-window memory truncation: masking alone
 * reproduces the observable behavior (sec.13), and at this model's scale
 * storing the full history is free. */

typedef struct {
    int len;
    float *k, *v;                 /* [cap, kv_dim] */
    float *k_hist, *v_hist;       /* [K-1, kv_dim] */
    float *attn_hist, *mlp_hist;  /* [K-1, hidden] */
} LayerCache;

typedef struct {
    int cap;
    int num_layers;
    LayerCache *layers;
} Cache;

static Cache *cache_create(const Model *m, int cap) {
    Cache *c = xmalloc(sizeof(Cache));
    c->cap = cap;
    c->num_layers = m->cfg.num_hidden_layers;
    c->layers = xcalloc((size_t)c->num_layers, sizeof(LayerCache));
    int Km1 = m->cfg.conv_kernel_size - 1;
    int hidden = m->cfg.hidden_size;
    for (int i = 0; i < c->num_layers; i++) {
        LayerCache *lc = &c->layers[i];
        int kv_dim = m->layers[i].kv_dim;
        lc->k = xcalloc((size_t)cap * (size_t)kv_dim, sizeof(float));
        lc->v = xcalloc((size_t)cap * (size_t)kv_dim, sizeof(float));
        lc->k_hist = xcalloc((size_t)Km1 * (size_t)kv_dim, sizeof(float));
        lc->v_hist = xcalloc((size_t)Km1 * (size_t)kv_dim, sizeof(float));
        lc->attn_hist = xcalloc((size_t)Km1 * (size_t)hidden, sizeof(float));
        lc->mlp_hist = xcalloc((size_t)Km1 * (size_t)hidden, sizeof(float));
    }
    return c;
}

static void cache_free(Cache *c) {
    for (int i = 0; i < c->num_layers; i++) {
        LayerCache *lc = &c->layers[i];
        free(lc->k);
        free(lc->v);
        free(lc->k_hist);
        free(lc->v_hist);
        free(lc->attn_hist);
        free(lc->mlp_hist);
    }
    free(c->layers);
    free(c);
}

/* =============================== attention =============================== */
/* Computes one decoder layer's self-attention block for T new positions
 * starting at absolute position start_pos, appending to the layer's KV
 * cache. Implements sec.2 (projections, per-head RMSNorm, GQA, the
 * 1/head_dim scaling), sec.3 (banded content-dependent relative position
 * bias) and sec.4 (log-scaling, global layers only) in one pass per query
 * position: causal/sliding-window masking is enforced by only ever looping
 * over the valid kv range, which is bit-identical to full-range plus an
 * additive -inf mask (excluded terms would contribute exactly 0.0 to the
 * softmax sum either way) while being far simpler to write correctly. */
static void attn_forward_chunk(const Config *cfg, const LayerWeights *lw, LayerCache *lc,
                                const float *x_normed /*[T,hidden]*/, int T, int start_pos,
                                float *out /*[T,hidden]*/) {
    int hidden = cfg->hidden_size;
    int H = lw->num_heads, Hkv = lw->num_kv_heads, Dh = lw->head_dim;
    int q_dim = lw->q_dim, kv_dim = lw->kv_dim, r_dim = lw->r_dim;
    int group = H / Hkv;
    int K = cfg->conv_kernel_size;
    int d_rel = cfg->d_rel;

    float *q_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    float *k_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *r_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)r_dim);
    for (int t = 0; t < T; t++) {
        const float *xt = x_normed + (size_t)t * hidden;
        linear(lw->wq, q_dim, hidden, xt, q_raw + (size_t)t * q_dim);
        linear(lw->wk, kv_dim, hidden, xt, k_raw + (size_t)t * kv_dim);
        linear(lw->wv, kv_dim, hidden, xt, v_raw + (size_t)t * kv_dim);
        linear(lw->wr, r_dim, hidden, xt, r_raw + (size_t)t * r_dim);
    }

    float *k_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    sconv_apply(lw->k_sconv_w, kv_dim, K, lc->k_hist, k_raw, T, k_sconv);
    sconv_apply(lw->v_sconv_w, kv_dim, K, lc->v_hist, v_raw, T, v_sconv);
    free(k_raw);
    free(v_raw);

    /* per-head RMSNorm: q_norm/k_norm applied after the view into
     * [...,heads,head_dim] but before transpose -- order only matters for
     * which axis is reduced (always head_dim), so a flat per-head-slice
     * normalize is exactly equivalent (sec.2). No norm on v. */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            float *qh = q_raw + (size_t)t * q_dim + (size_t)h * Dh;
            rmsnorm(qh, lw->q_norm, Dh, (float)cfg->rms_norm_eps, qh);
        }
        for (int h = 0; h < Hkv; h++) {
            float *kh = k_sconv + (size_t)t * kv_dim + (size_t)h * Dh;
            rmsnorm(kh, lw->k_norm, Dh, (float)cfg->rms_norm_eps, kh);
        }
    }

    for (int t = 0; t < T; t++) {
        memcpy(lc->k + (size_t)(start_pos + t) * kv_dim, k_sconv + (size_t)t * kv_dim, sizeof(float) * (size_t)kv_dim);
        memcpy(lc->v + (size_t)(start_pos + t) * kv_dim, v_sconv + (size_t)t * kv_dim, sizeof(float) * (size_t)kv_dim);
    }
    if (start_pos + T > lc->len) lc->len = start_pos + T;
    free(k_sconv);
    free(v_sconv);

    int have_log_scaling = (!lw->is_sliding) && cfg->has_log_scaling_floor; /* local layers never apply this (sec.4) */

    float *attn_concat = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim); /* [T, H*Dh], pre-o_proj */
    float *rel_logits = xmalloc(sizeof(float) * (size_t)lw->rel_extent);
    float *scores = xmalloc(sizeof(float) * (size_t)(start_pos + T)); /* upper bound on any kv range length */
    float *q_scaled = xmalloc(sizeof(float) * (size_t)Dh);
    double *accd = xmalloc(sizeof(double) * (size_t)Dh);

    for (int t = 0; t < T; t++) {
        int q_pos = start_pos + t;
        double tau = 1.0;
        if (have_log_scaling) {
            double effective_n = (double)(q_pos + 1);
            double ratio = effective_n / (double)cfg->log_scaling_n_floor;
            if (ratio < 1.0) ratio = 1.0;
            tau = 1.0 + cfg->log_scaling_alpha * log(ratio); /* untested by the toy oracle, sec.4/sec.13 */
        }

        int kv_lo = lw->is_sliding ? (q_pos - cfg->sliding_window_size + 1) : 0;
        if (kv_lo < 0) kv_lo = 0;
        int kv_hi = q_pos; /* inclusive; causal */
        int n_kv = kv_hi - kv_lo + 1;

        for (int h = 0; h < H; h++) {
            int hk = h / group;
            const float *r_vec = r_raw + (size_t)t * r_dim + (size_t)h * d_rel;
            /* rel_logits[r] = sum_d r_vec[d] * proj[d, r]; proj is shared
             * across heads within a layer (einsum broadcasts it, sec.3). */
            for (int r = 0; r < lw->rel_extent; r++) {
                double acc = 0.0;
                for (int d = 0; d < d_rel; d++) acc += (double)r_vec[d] * (double)lw->rel_proj[(size_t)d * lw->rel_extent + r];
                rel_logits[r] = (float)acc;
            }

            const float *q_vec = q_raw + (size_t)t * q_dim + (size_t)h * Dh;
            for (int d = 0; d < Dh; d++) q_scaled[d] = (float)((double)q_vec[d] * tau);

            float max_score = -INFINITY;
            for (int kv = kv_lo; kv <= kv_hi; kv++) {
                const float *k_vec = lc->k + (size_t)kv * kv_dim + (size_t)hk * Dh;
                double content = (double)dotf(q_scaled, k_vec, Dh) * (1.0 / (double)Dh); /* scaling = 1/head_dim, not 1/sqrt (sec.2) */
                int distance = q_pos - kv;
                double bias = 0.0;
                if (distance < lw->rel_extent) bias = (double)rel_logits[distance] * tau; /* clamp/mask, sec.3 */
                float sc = (float)(content + bias);
                scores[kv - kv_lo] = sc;
                if (sc > max_score) max_score = sc;
            }
            double sum_exp = 0.0;
            for (int i = 0; i < n_kv; i++) {
                double e = exp((double)(scores[i] - max_score));
                scores[i] = (float)e;
                sum_exp += e;
            }
            for (int d = 0; d < Dh; d++) accd[d] = 0.0;
            for (int i = 0; i < n_kv; i++) {
                int kv = kv_lo + i;
                double wgt = (double)scores[i] / sum_exp;
                const float *v_vec = lc->v + (size_t)kv * kv_dim + (size_t)hk * Dh;
                for (int d = 0; d < Dh; d++) accd[d] += wgt * (double)v_vec[d];
            }
            float *outh = attn_concat + (size_t)t * q_dim + (size_t)h * Dh;
            for (int d = 0; d < Dh; d++) outh[d] = (float)accd[d];
        }
    }
    free(rel_logits);
    free(scores);
    free(q_scaled);
    free(accd);
    free(q_raw);
    free(r_raw);

    for (int t = 0; t < T; t++) linear(lw->wo, hidden, q_dim, attn_concat + (size_t)t * q_dim, out + (size_t)t * hidden);
    free(attn_concat);
}

/* ================================== MLP ================================= */
/* Dense SwiGLU FFN (layers < dense_mlp_idx). global_scale multiplies the
 * FFN *output* here (sec.9) -- contrast with the MoE router below, where
 * the analogous per-layer scale multiplies the routing *weights* instead. */
static void mlp_dense_forward(const LayerWeights *lw, int hidden, int dense_inter, const float *x, float *out) {
    float *h = xmalloc(sizeof(float) * (size_t)dense_inter);
    int two_inter = 2 * dense_inter;
    for (int i = 0; i < dense_inter; i++) {
        float g = dotf(w13_row(lw->dense_w13, 0, two_inter, hidden, 0, i), x, hidden);
        float u = dotf(w13_row(lw->dense_w13, 0, two_inter, hidden, 1, i), x, hidden);
        h[i] = silu_f(g) * u;
    }
    float gscale = lw->dense_global_scale[0];
    for (int d = 0; d < hidden; d++) {
        const float *row = lw->dense_w2 + (size_t)d * dense_inter;
        out[d] = dotf(row, h, dense_inter) * gscale;
    }
    free(h);
}

/* Router (sec.6) + routed experts (sec.7) + shared experts (sec.7). The
 * single most error-prone part per architecture-notes.md: the aux-loss-free
 * bias (router_bias) selects experts (topk) but is NOT part of the mixing
 * weight, which comes from log_sigmoid/logsumexp over the RAW logits of the
 * selected routed experts plus the always-on shared-expert-sink logits. */
static void mlp_moe_forward(const Config *cfg, const LayerWeights *lw, const float *x, float *out) {
    int hidden = cfg->hidden_size;
    int n_routed = cfg->n_routed_experts, n_shared = cfg->n_shared_experts, topk = cfg->num_experts_per_tok;
    int n_total = n_routed + n_shared;
    int moe_inter = cfg->moe_intermediate_size;
    int two_inter = 2 * moe_inter;

    float *router_logits = xmalloc(sizeof(float) * (size_t)n_total);
    for (int i = 0; i < n_total; i++) router_logits[i] = dotf(lw->router_w + (size_t)i * hidden, x, hidden);

    float *scores_for_choice = xmalloc(sizeof(float) * (size_t)n_routed);
    for (int i = 0; i < n_routed; i++) scores_for_choice[i] = sigmoid_f(router_logits[i]) + lw->router_bias[i];

    int *topk_idx = xmalloc(sizeof(int) * (size_t)topk);
    {
        int *used = xcalloc((size_t)n_routed, sizeof(int));
        for (int j = 0; j < topk; j++) {
            int best = -1;
            for (int i = 0; i < n_routed; i++) {
                if (used[i]) continue;
                if (best == -1 || scores_for_choice[i] > scores_for_choice[best]) best = i;
            }
            topk_idx[j] = best;
            used[best] = 1;
        }
        free(used);
    }

    int n_sel = topk + n_shared;
    float *topk_logits = xmalloc(sizeof(float) * (size_t)n_sel);
    for (int j = 0; j < topk; j++) topk_logits[j] = router_logits[topk_idx[j]]; /* raw, pre-sigmoid, pre-bias */
    for (int s = 0; s < n_shared; s++) topk_logits[topk + s] = router_logits[n_routed + s];

    float *log_probs = xmalloc(sizeof(float) * (size_t)n_sel);
    for (int j = 0; j < n_sel; j++) log_probs[j] = logsigmoid_f(topk_logits[j]);

    float lmax = log_probs[0];
    for (int j = 1; j < n_sel; j++)
        if (log_probs[j] > lmax) lmax = log_probs[j];
    double sumexp = 0.0;
    for (int j = 0; j < n_sel; j++) sumexp += exp((double)(log_probs[j] - lmax));
    double lse = lmax + log(sumexp);

    float *weights = xmalloc(sizeof(float) * (size_t)n_sel);
    double gs = (double)lw->router_global_scale[0];
    for (int j = 0; j < n_sel; j++) {
        double w = exp((double)log_probs[j] - lse);
        w *= cfg->route_scale * gs; /* norm_after_topk's renormalization, then route_scale*global_scale (sec.6) */
        weights[j] = (float)w;
    }

    for (int d = 0; d < hidden; d++) out[d] = 0.0f;

    float *h = xmalloc(sizeof(float) * (size_t)moe_inter);
    float *expert_out = xmalloc(sizeof(float) * (size_t)hidden);
    for (int j = 0; j < topk; j++) {
        int e = topk_idx[j];
        if (lw->real_exps) {
            /* Task 14: streamed quantized weights (pread + qlinear), same
             * silu/gating math as the in-memory branch below. */
            real_expert_ffn(lw->real_exps, e, x, expert_out);
        } else {
            for (int i = 0; i < moe_inter; i++) {
                float g = dotf(w13_row(lw->experts_w13, e, two_inter, hidden, 0, i), x, hidden);
                float u = dotf(w13_row(lw->experts_w13, e, two_inter, hidden, 1, i), x, hidden);
                h[i] = silu_f(g) * u;
            }
            for (int d = 0; d < hidden; d++) {
                const float *row = lw->experts_w2 + ((size_t)e * hidden + (size_t)d) * moe_inter;
                expert_out[d] = dotf(row, h, moe_inter);
            }
        }
        float wj = weights[j];
        for (int d = 0; d < hidden; d++) out[d] += expert_out[d] * wj; /* routed: weight applied AFTER down_proj (sec.7) */
    }
    free(expert_out);

    for (int s = 0; s < n_shared; s++) {
        float gamma = weights[topk + s];
        for (int i = 0; i < moe_inter; i++) {
            float g = dotf(w13_row(lw->shared_w13, s, two_inter, hidden, 0, i), x, hidden);
            float u = dotf(w13_row(lw->shared_w13, s, two_inter, hidden, 1, i), x, hidden);
            h[i] = silu_f(g) * u * gamma; /* shared: gamma multiplies the activation, here, matching
                                            * InklingSharedExperts.forward's literal placement -- down_proj
                                            * is linear, so this is mathematically equivalent to scaling its
                                            * output by gamma instead (as the routed path above does with
                                            * its per-expert weight); not a semantic difference from routed,
                                            * just matching the reference's order of operations exactly
                                            * (sec.7) */
        }
        for (int d = 0; d < hidden; d++) {
            const float *row = lw->shared_w2 + ((size_t)s * hidden + (size_t)d) * moe_inter;
            out[d] += dotf(row, h, moe_inter);
        }
    }

    free(h);
    free(weights);
    free(log_probs);
    free(topk_logits);
    free(topk_idx);
    free(scores_for_choice);
    free(router_logits);
}

/* real_expert_ffn's full definition lives in the "real model" section below,
 * after RealExperts is defined -- the forward declaration near LayerWeights
 * (and the seam call site in mlp_moe_forward above) is all that's needed
 * here, since real_exps is always NULL in oracle mode (this branch is never
 * taken by the self-test). */

/* ============================= decoder layer ============================= */
/* Pre-norm residual wiring (sec.0/InklingDecoderLayer.forward, verified
 * directly): attention and MLP/MoE each get their own outer residual, and
 * each block's raw output additionally passes through its own sconv (with
 * its own INTERNAL residual, sec.5) before joining the outer residual. */
static void decoder_layer_forward(const Config *cfg, const LayerWeights *lw, LayerCache *lc, float *x /*[T,hidden], in/out*/,
                                   int T, int start_pos) {
    int hidden = cfg->hidden_size;
    int K = cfg->conv_kernel_size;

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) rmsnorm(x + (size_t)t * hidden, lw->attn_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);

    float *attn_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    attn_forward_chunk(cfg, lw, lc, normed, T, start_pos, attn_out);

    float *attn_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    sconv_apply(lw->attn_sconv_w, hidden, K, lc->attn_hist, attn_out, T, attn_sconv_out);
    for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += attn_sconv_out[i];
    free(attn_out);
    free(attn_sconv_out);

    for (int t = 0; t < T; t++) rmsnorm(x + (size_t)t * hidden, lw->mlp_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);

    float *mlp_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        if (lw->is_sparse)
            mlp_moe_forward(cfg, lw, normed + (size_t)t * hidden, mlp_out + (size_t)t * hidden);
        else
            mlp_dense_forward(lw, hidden, cfg->dense_intermediate_size, normed + (size_t)t * hidden, mlp_out + (size_t)t * hidden);
    }
    free(normed);

    float *mlp_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    sconv_apply(lw->mlp_sconv_w, hidden, K, lc->mlp_hist, mlp_out, T, mlp_sconv_out);
    for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += mlp_sconv_out[i];
    free(mlp_out);
    free(mlp_sconv_out);
}

/* ========================== activation dump (debug) ======================= */
/* `--dump-acts FILE` writes a simple stream of records (namelen, name,
 * ndim, dims, float32 data), read back and diffed against the Python
 * reference by tools/dump_activations.py --compare. Not npz (no zip writer
 * needed); the pairing is this file plus that script, not a general format. */

typedef struct {
    FILE *f;
} ActDump;

static ActDump dump_open(const char *path) {
    ActDump d;
    d.f = fopen(path, "wb");
    if (!d.f) die("open %s for writing: %s", path, strerror(errno));
    return d;
}

static void dump_capture(ActDump *d, const char *name, const float *data, int T, int C) {
    uint32_t namelen = (uint32_t)strlen(name);
    fwrite(&namelen, sizeof namelen, 1, d->f);
    fwrite(name, 1, namelen, d->f);
    uint32_t ndim = 2;
    fwrite(&ndim, sizeof ndim, 1, d->f);
    uint32_t dims[2] = {(uint32_t)T, (uint32_t)C};
    fwrite(dims, sizeof(uint32_t), 2, d->f);
    fwrite(data, sizeof(float), (size_t)T * (size_t)C, d->f);
}

static void dump_close(ActDump *d) { fclose(d->f); }

/* ============================== top-level forward ========================= */
/* Processes T new tokens starting at absolute position start_pos against
 * `cache` (already primed with start_pos prior positions, or empty for a
 * from-scratch chunk). The same function serves the self-test's
 * cache-less 32-position teacher-forcing pass (T=32, start_pos=0, a cache
 * created and discarded for that one call) and the incremental
 * prompt-priming + decode loop (T=12 once, then T=1 repeatedly against a
 * cache that persists across calls) -- sec.11 confirms these reduce to the
 * same math for a from-scratch chunk. logits_out and dump are both
 * optional (NULL to skip). */
static void model_forward_chunk(const Model *m, Cache *cache, const int *token_ids, int T, int start_pos,
                                 float *logits_out /*[T,unpadded_vocab] or NULL*/, ActDump *dump) {
    const Config *cfg = &m->cfg;
    int hidden = cfg->hidden_size;

    float *x = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        const float *erow = m->embed + (size_t)token_ids[t] * hidden;
        rmsnorm(erow, m->embed_norm, hidden, (float)cfg->rms_norm_eps, x + (size_t)t * hidden);
    }
    if (dump) dump_capture(dump, "embed_out", x, T, hidden);

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        decoder_layer_forward(cfg, &m->layers[l], &cache->layers[l], x, T, start_pos);
        if (dump) {
            char name[32];
            snprintf(name, sizeof name, "layer%d.out", l);
            dump_capture(dump, name, x, T, hidden);
        }
    }

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) rmsnorm(x + (size_t)t * hidden, m->final_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
    if (dump) dump_capture(dump, "final_norm_out", normed, T, hidden);
    free(x);

    if (logits_out) {
        int unpadded = cfg->unpadded_vocab_size;
        float *h = xmalloc(sizeof(float) * (size_t)hidden);
        float mup = (float)cfg->logits_mup_width_multiplier;
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < hidden; d++) h[d] = normed[(size_t)t * hidden + d] / mup;
            for (int v = 0; v < unpadded; v++)
                logits_out[(size_t)t * unpadded + v] = dotf(m->unembed + (size_t)v * hidden, h, hidden);
        }
        free(h);
        if (dump) dump_capture(dump, "logits", logits_out, T, unpadded);
    }
    free(normed);
}

/* ============================ oracle reference ============================ */

typedef struct {
    int *ids;
    int len;
} IntArr;

static IntArr json_int_array(const JsonValue *v, const char *what) {
    if (!v || v->type != JSON_ARR) die("ref_inkling.json: expected array for %s", what);
    IntArr a;
    a.len = (int)v->arr_count;
    a.ids = xmalloc(sizeof(int) * (size_t)a.len);
    for (int i = 0; i < a.len; i++) a.ids[i] = (int)json_num(v->arr_items[i], 0);
    return a;
}

typedef struct {
    IntArr prompt_ids;
    IntArr full_ids;
    IntArr tf_pred;
} OracleRef;

static OracleRef load_oracle_ref(const char *path) {
    size_t len;
    char *buf = read_file(path, &len);
    JsonValue *root = json_parse(buf, len);
    OracleRef r;
    r.prompt_ids = json_int_array(json_get(root, "prompt_ids"), "prompt_ids");
    r.full_ids = json_int_array(json_get(root, "full_ids"), "full_ids");
    r.tf_pred = json_int_array(json_get(root, "tf_pred"), "tf_pred");
    free(buf);
    return r;
}

/* ================================ self-test ================================ */

static const char *CONFIG_PATH = "tools/oracle/tiny/config.json";
static const char *WEIGHTS_PATH = "tools/oracle/tiny/model.safetensors";
static const char *REF_PATH = "tools/oracle/ref_inkling.json";

static void report_mismatch(const char *phase, int pos, int expected, int got, const float *logits, int n) {
    fprintf(stderr, "%s mismatch at position %d: expected=%d got=%d\n", phase, pos, expected, got);
    int idx[3] = {-1, -1, -1};
    float val[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < n; i++) {
        float v = logits[i];
        if (v > val[0]) {
            val[2] = val[1]; idx[2] = idx[1];
            val[1] = val[0]; idx[1] = idx[0];
            val[0] = v; idx[0] = i;
        } else if (v > val[1]) {
            val[2] = val[1]; idx[2] = idx[1];
            val[1] = v; idx[1] = i;
        } else if (v > val[2]) {
            val[2] = v; idx[2] = i;
        }
    }
    fprintf(stderr, "  top3: [%d]=%.6f  [%d]=%.6f  [%d]=%.6f\n", idx[0], val[0], idx[1], val[1], idx[2], val[2]);
}

/* Gate: prefill (cache-less teacher forcing over all full_ids, argmax at
 * every position) and decode (true autoregressive greedy generation from
 * prompt_ids, KV cache + sconv state updated incrementally) each reported
 * separately, per the brief. Returns 1 iff both hit a perfect score. */
static int run_self_test(const Model *m, const char *ref_path) {
    OracleRef ref = load_oracle_ref(ref_path);
    int unpadded = m->cfg.unpadded_vocab_size;
    int full_len = ref.full_ids.len;
    int prompt_len = ref.prompt_ids.len;

    /* --- prefill / teacher forcing --- */
    Cache *tf_cache = cache_create(m, full_len);
    float *logits = xmalloc(sizeof(float) * (size_t)full_len * (size_t)unpadded);
    model_forward_chunk(m, tf_cache, ref.full_ids.ids, full_len, 0, logits, NULL);
    cache_free(tf_cache);

    int prefill_ok = 0;
    for (int t = 0; t < full_len; t++) {
        const float *lt = logits + (size_t)t * unpadded;
        int got = argmax_f(lt, unpadded);
        int expected = ref.tf_pred.ids[t];
        if (got == expected)
            prefill_ok++;
        else
            report_mismatch("prefill", t, expected, got, lt, unpadded);
    }
    printf("prefill %d/%d\n", prefill_ok, full_len);
    free(logits);

    /* --- incremental decode --- */
    Cache *dc = cache_create(m, full_len);
    float *plogits = xmalloc(sizeof(float) * (size_t)prompt_len * (size_t)unpadded);
    model_forward_chunk(m, dc, ref.prompt_ids.ids, prompt_len, 0, plogits, NULL);

    float *cur_logits = xmalloc(sizeof(float) * (size_t)unpadded);
    memcpy(cur_logits, plogits + (size_t)(prompt_len - 1) * unpadded, sizeof(float) * (size_t)unpadded);
    free(plogits);

    int n_decode = full_len - prompt_len;
    int decode_ok = 0;
    int cur_pos = prompt_len;
    for (int i = 0; i < n_decode; i++) {
        int got = argmax_f(cur_logits, unpadded);
        int expected = ref.full_ids.ids[prompt_len + i];
        if (got == expected)
            decode_ok++;
        else
            report_mismatch("decode", i, expected, got, cur_logits, unpadded);
        int next_id = got; /* true greedy generation: feed our own prediction forward */
        model_forward_chunk(m, dc, &next_id, 1, cur_pos, cur_logits, NULL);
        cur_pos++;
    }
    printf("decode %d/%d\n", decode_ok, n_decode);
    free(cur_logits);
    cache_free(dc);

    return (prefill_ok == full_len) && (decode_ok == n_decode);
}

/* ============================== real model (Task 14) ====================== */
/* Real-checkpoint loader + streaming MoE forward. Design principle: reuse
 * the Phase-0-verified forward math (attn_forward_chunk, mlp_dense_forward,
 * mlp_moe_forward, decoder_layer_forward) completely unchanged. Real mode
 * only supplies a *different way to fill LayerWeights* each layer -- F32
 * tensors point straight into the resident.bin mmap, quantized tensors are
 * dequantized row-by-row into a reusable f32 arena (qrow), and the routed
 * (non-resident) experts are streamed via the pluggable seam added in
 * Step 1 (real_expert_ffn). See docs/container.md sec.2-3 and the tensor
 * mapping table in .superpowers/sdd/task-14-brief.md for the on-disk
 * conventions this section implements.
 *
 * GGUF shape convention (verified directly against the real sidecars,
 * cross-checked against docs/gguf-inventory-ud-q2_k_xl.md's per-expert
 * table): for an N-d tensor, shape[0] is the fastest-varying/contiguous
 * dimension and shape[N-1] the slowest. For every 2D weight matrix this
 * engine consumes, that means in_dim=shape[0] (contiguous per row),
 * out_dim=shape[1] (row count) -- exactly the QTensor{out_dim,in_dim}
 * convention already used above. For the 3D "stacked" shared-expert
 * tensors (ffn_{gate,up,down}_shexp.weight), shape[2] is the stack count
 * (2 shared experts), and each stack's bytes are a fully contiguous 2D
 * matrix in the same [out_dim,in_dim] layout, back to back.
 */

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = xmalloc(n);
    memcpy(p, s, n);
    return p;
}

/* dirname(path) + "/" + suffix. Used to derive the GGUF parts directory
 * from the index sidecar's own path (weights/inkling-gguf sits alongside
 * weights/inkling-ud-q2_k_xl.sepia-index.json) and resident.bin's path
 * from the manifest's, without needing extra CLI flags -- the same
 * convention serves both --real and --smoke. */
static char *dirname_join(const char *path, const char *suffix) {
    const char *slash = strrchr(path, '/');
    const char *dir = slash ? path : ".";
    size_t dirlen = slash ? (size_t)(slash - path) : 1;
    size_t n = dirlen + 1 + strlen(suffix) + 1;
    char *out = xmalloc(n);
    memcpy(out, dir, dirlen);
    out[dirlen] = '/';
    memcpy(out + dirlen + 1, suffix, strlen(suffix) + 1);
    return out;
}

/* Generic (not text_config-scoped) required-field JSON accessors, for the
 * manifest/index sidecars -- config.json's own json_get_int/json_get_double
 * above are scoped to a specific "text_config" object and don't fit here. */
static int64_t json_req_i64(const JsonValue *obj, const char *key, const char *what) {
    const JsonValue *v = json_get(obj, key);
    if (json_is_null(v)) die("%s: missing required field '%s'", what, key);
    return (int64_t)json_num(v, 0);
}
static int json_req_int(const JsonValue *obj, const char *key, const char *what) {
    return (int)json_req_i64(obj, key, what);
}
static const char *json_req_str(const JsonValue *obj, const char *key, const char *what) {
    const JsonValue *v = json_get(obj, key);
    const char *s = json_str(v, NULL);
    if (!s) die("%s: missing required string field '%s'", what, key);
    return s;
}

/* ------------------------- resident-manifest.json -------------------------- */

typedef struct {
    char *name;
    int ggml_type;
    int64_t shape[4];
    int n_dims;
    int64_t offset; /* into res_base */
    int64_t nbytes;
} ResidentEntry;

typedef struct {
    ResidentEntry *entries;
    size_t count;
    const uint8_t *res_base;
    size_t res_size;
} ResidentTable;

/* Parses resident-manifest.json's "tensors" array into a name-indexed table
 * (linear array + strcmp lookup -- ~1300 tensors, load-time only, per the
 * brief). Every entry's [offset,offset+nbytes) is bounds-checked against
 * res_size, same discipline as st_find's safetensors bounds check above. */
static ResidentTable manifest_load(const char *path, const void *res_base, size_t res_size) {
    size_t len;
    char *buf = read_file(path, &len);
    JsonValue *root = json_parse(buf, len);
    const JsonValue *tensors = json_get(root, "tensors");
    if (!tensors || tensors->type != JSON_ARR) die("%s: missing 'tensors' array", path);

    ResidentTable rt = {0};
    rt.res_base = (const uint8_t *)res_base;
    rt.res_size = res_size;
    rt.count = tensors->arr_count;
    rt.entries = xcalloc(rt.count, sizeof(ResidentEntry));
    for (size_t i = 0; i < rt.count; i++) {
        const JsonValue *t = tensors->arr_items[i];
        ResidentEntry *e = &rt.entries[i];
        e->name = xstrdup(json_req_str(t, "name", path));
        e->ggml_type = json_req_int(t, "ggml_type", path);
        e->offset = json_req_i64(t, "offset", path);
        e->nbytes = json_req_i64(t, "nbytes", path);
        const JsonValue *shape = json_get(t, "shape");
        if (!shape || shape->type != JSON_ARR || shape->arr_count == 0 || shape->arr_count > 4)
            die("%s: tensor '%s' has a missing/invalid 'shape' (1-4 dims expected)", path, e->name);
        e->n_dims = (int)shape->arr_count;
        for (int d = 0; d < e->n_dims; d++) e->shape[d] = (int64_t)json_num(shape->arr_items[d], 0);
        if (e->offset < 0 || e->nbytes < 0 || (uint64_t)e->offset + (uint64_t)e->nbytes > (uint64_t)res_size)
            die("%s: tensor '%s' [%lld,%lld) exceeds resident.bin size %zu bytes",
                path, e->name, (long long)e->offset, (long long)(e->offset + e->nbytes), res_size);
    }
    free(buf);
    return rt;
}

static const ResidentEntry *resident_find(const ResidentTable *rt, const char *name) {
    for (size_t i = 0; i < rt->count; i++)
        if (strcmp(rt->entries[i].name, name) == 0) return &rt->entries[i];
    return NULL;
}
static const ResidentEntry *resident_get(const ResidentTable *rt, const char *name) {
    const ResidentEntry *e = resident_find(rt, name);
    if (!e) die("resident-manifest: missing tensor '%s'", name);
    return e;
}

/* F32 resident tensor: direct pointer into res_base, no dequant/copy. */
static const float *resident_f32(const ResidentTable *rt, const char *name) {
    const ResidentEntry *e = resident_get(rt, name);
    if (e->ggml_type != SEPIA_T_F32) die("resident tensor '%s': expected F32, got ggml_type %d", name, e->ggml_type);
    int64_t numel = 1;
    for (int d = 0; d < e->n_dims; d++) numel *= e->shape[d];
    if (e->nbytes != numel * (int64_t)sizeof(float))
        die("resident tensor '%s': nbytes %lld disagrees with shape (numel=%lld)",
            name, (long long)e->nbytes, (long long)numel);
    return (const float *)(rt->res_base + e->offset);
}

/* 2D weight matrix -> QTensor{out_dim=shape[1], in_dim=shape[0]} (see the
 * shape-convention note at the top of this section). */
static QTensor resident_qtensor(const ResidentTable *rt, const char *name) {
    const ResidentEntry *e = resident_get(rt, name);
    if (e->n_dims != 2) die("resident tensor '%s': expected 2 dims, got %d", name, e->n_dims);
    QTensor q;
    q.ggml_type = e->ggml_type;
    q.data = rt->res_base + e->offset;
    q.in_dim = e->shape[0];
    q.out_dim = e->shape[1];
    size_t want = quants_row_bytes(e->ggml_type, q.in_dim) * (size_t)q.out_dim;
    if ((size_t)e->nbytes != want)
        die("resident tensor '%s': nbytes %lld disagrees with shape/type (want %zu)",
            name, (long long)e->nbytes, want);
    return q;
}

/* 3D tensor with n_stack "experts" stacked along the slowest-varying axis
 * (shared-expert gate/up/down: ffn_{gate,up,down}_shexp.weight, 2 stacked).
 * Returns stack index 0's QTensor plus the per-stack byte stride; caller
 * offsets .data by s*stride for s in [1,n_stack). */
static QTensor resident_qtensor_stacked(const ResidentTable *rt, const char *name, int n_stack, int64_t *stride_out) {
    const ResidentEntry *e = resident_get(rt, name);
    if (e->n_dims != 3) die("resident tensor '%s': expected 3 dims (stacked), got %d", name, e->n_dims);
    if (e->shape[2] != n_stack)
        die("resident tensor '%s': stack dim %lld != expected %d", name, (long long)e->shape[2], n_stack);
    if (e->nbytes % n_stack != 0)
        die("resident tensor '%s': nbytes %lld not a multiple of %d stacks", name, (long long)e->nbytes, n_stack);
    QTensor q;
    q.ggml_type = e->ggml_type;
    q.data = rt->res_base + e->offset;
    q.in_dim = e->shape[0];
    q.out_dim = e->shape[1];
    int64_t stride = e->nbytes / n_stack;
    size_t want = quants_row_bytes(e->ggml_type, q.in_dim) * (size_t)q.out_dim;
    if ((size_t)stride != want)
        die("resident tensor '%s': per-stack bytes %lld disagrees with shape/type (want %zu)",
            name, (long long)stride, want);
    *stride_out = stride;
    return q;
}

/* Per-layer resident tensor names are "blk.<layer>.<suffix>", mirroring the
 * oracle loader's find_layer_tensor above. */
static const float *real_layer_f32(const ResidentTable *rt, int layer, const char *suffix) {
    char name[160];
    int n = snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof name) die("tensor name too long for layer %d suffix %s", layer, suffix);
    return resident_f32(rt, name);
}
static QTensor real_layer_qtensor(const ResidentTable *rt, int layer, const char *suffix) {
    char name[160];
    int n = snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof name) die("tensor name too long for layer %d suffix %s", layer, suffix);
    return resident_qtensor(rt, name);
}
static QTensor real_layer_qtensor_stacked(const ResidentTable *rt, int layer, const char *suffix, int n_stack, int64_t *stride_out) {
    char name[160];
    int n = snprintf(name, sizeof name, "blk.%d.%s", layer, suffix);
    if (n < 0 || (size_t)n >= sizeof name) die("tensor name too long for layer %d suffix %s", layer, suffix);
    return resident_qtensor_stacked(rt, name, n_stack, stride_out);
}

/* ------------------------- sepia-index.json (experts) ----------------------- */

typedef struct {
    int part_idx;
    int64_t abs_offset;
    int64_t nbytes;
    int ggml_type;
} ExpertSlot;

typedef struct {
    ExpertSlot *gate, *up, *down; /* [n_experts] each; all NULL if this layer has no MoE entry (dense layer) */
} MoeLayerIndex;

typedef struct {
    char **part_files; /* [n_parts], relative to dirname(index_path)/inkling-gguf */
    int n_parts;
    int n_experts;      /* n_experts_per_layer, read from the JSON (not hardcoded --
                          * the real deployment is 256, the smoke fixture is 2) */
    MoeLayerIndex *by_layer; /* [n_layers_alloc]; dense/missing layers left zeroed */
    int n_layers_alloc;
    int64_t max_slab_bytes; /* max nbytes over every gate/up/down/expert entry --
                              * sizes real_expert_ffn's reusable pread buffers */
} ExpertIndex;

static int index_find_part(const ExpertIndex *idx, const char *part_file) {
    for (int i = 0; i < idx->n_parts; i++)
        if (strcmp(idx->part_files[i], part_file) == 0) return i;
    return -1;
}

static void index_parse_slot(const JsonValue *slot_obj, const char *path, int layer, const char *slot_name,
                              ExpertIndex *idx, ExpertSlot *out /* [n_experts] */) {
    const JsonValue *experts = json_get(slot_obj, "experts");
    if (!experts || experts->type != JSON_ARR)
        die("%s: layer %d slot '%s' missing 'experts' array", path, layer, slot_name);
    if ((int)experts->arr_count != idx->n_experts)
        die("%s: layer %d slot '%s' has %zu experts, expected n_experts_per_layer=%d",
            path, layer, slot_name, experts->arr_count, idx->n_experts);
    for (int e = 0; e < idx->n_experts; e++) {
        const JsonValue *ent = experts->arr_items[e];
        const char *part_file = json_req_str(ent, "part_file", path);
        int part_idx = index_find_part(idx, part_file);
        if (part_idx < 0)
            die("%s: layer %d slot '%s' expert %d references unknown part_file '%s'", path, layer, slot_name, e, part_file);
        int64_t abs_offset = json_req_i64(ent, "abs_offset", path);
        int64_t nbytes = json_req_i64(ent, "nbytes", path);
        int ggml_type = json_req_int(ent, "ggml_type", path);
        if (abs_offset < 0 || nbytes < 0)
            die("%s: layer %d slot '%s' expert %d has a negative offset/nbytes", path, layer, slot_name, e);
        out[e].part_idx = part_idx;
        out[e].abs_offset = abs_offset;
        out[e].nbytes = nbytes;
        out[e].ggml_type = ggml_type;
        if (nbytes > idx->max_slab_bytes) idx->max_slab_bytes = nbytes;
    }
}

/* Parses sepia-index.json's parts[] file list and moe_layers{} object.
 * Dies if a layer is missing any of gate/up/down, if an "experts" array's
 * length disagrees with the top-level n_experts_per_layer, or if more than
 * 8 parts are listed (RealModel.part_fds is fixed-size 8). */
static ExpertIndex index_load(const char *path) {
    size_t len;
    char *buf = read_file(path, &len);
    JsonValue *root = json_parse(buf, len);

    ExpertIndex idx = {0};
    idx.n_experts = json_req_int(root, "n_experts_per_layer", path);
    if (idx.n_experts <= 0) die("%s: n_experts_per_layer must be positive, got %d", path, idx.n_experts);

    const JsonValue *parts = json_get(root, "parts");
    if (!parts || parts->type != JSON_ARR) die("%s: missing 'parts' array", path);
    idx.n_parts = (int)parts->arr_count;
    if (idx.n_parts <= 0 || idx.n_parts > 8) die("%s: 'parts' has %d entries, expected 1-8", path, idx.n_parts);
    idx.part_files = xcalloc((size_t)idx.n_parts, sizeof(char *));
    for (int i = 0; i < idx.n_parts; i++)
        idx.part_files[i] = xstrdup(json_req_str(parts->arr_items[i], "part_file", path));

    const JsonValue *moe_layers = json_get(root, "moe_layers");
    if (!moe_layers || moe_layers->type != JSON_OBJ) die("%s: missing 'moe_layers' object", path);

    int max_layer = -1;
    for (size_t i = 0; i < moe_layers->obj_count; i++) {
        int layer = atoi(moe_layers->obj_keys[i]);
        if (layer < 0) die("%s: bad moe_layers key '%s'", path, moe_layers->obj_keys[i]);
        if (layer > max_layer) max_layer = layer;
    }
    idx.n_layers_alloc = max_layer + 1;
    idx.by_layer = xcalloc((size_t)idx.n_layers_alloc, sizeof(MoeLayerIndex));

    for (size_t i = 0; i < moe_layers->obj_count; i++) {
        int layer = atoi(moe_layers->obj_keys[i]);
        const JsonValue *kinds = moe_layers->obj_vals[i];
        MoeLayerIndex *mli = &idx.by_layer[layer];
        mli->gate = xcalloc((size_t)idx.n_experts, sizeof(ExpertSlot));
        mli->up   = xcalloc((size_t)idx.n_experts, sizeof(ExpertSlot));
        mli->down = xcalloc((size_t)idx.n_experts, sizeof(ExpertSlot));
        const JsonValue *gate_obj = json_get(kinds, "gate");
        const JsonValue *up_obj   = json_get(kinds, "up");
        const JsonValue *down_obj = json_get(kinds, "down");
        if (!gate_obj || !up_obj || !down_obj)
            die("%s: layer %d is missing gate/up/down (found gate=%d up=%d down=%d)",
                path, layer, gate_obj != NULL, up_obj != NULL, down_obj != NULL);
        index_parse_slot(gate_obj, path, layer, "gate", &idx, mli->gate);
        index_parse_slot(up_obj,   path, layer, "up",   &idx, mli->up);
        index_parse_slot(down_obj, path, layer, "down", &idx, mli->down);
    }

    free(buf);
    return idx;
}

/* ---------------------------- real model structs ---------------------------- */

/* Per-layer resolved lookups (F32 direct pointers + QTensor descriptors),
 * resolved once at real_load time by strcmp against the manifest -- NOT
 * re-resolved on every forward call. real_fill_layer only dequantizes from
 * these already-resolved descriptors into the per-call arena. */
typedef struct {
    const float *attn_norm, *mlp_norm, *q_norm, *k_norm, *rel_proj;
    const float *k_sconv_w, *v_sconv_w, *attn_sconv_w, *mlp_sconv_w;

    QTensor wq, wk, wv, wr, wo;

    /* dense only */
    const float *dense_global_scale;
    QTensor dense_gate, dense_up, dense_w2;

    /* sparse only */
    const float *router_w, *router_bias, *router_global_scale;
    QTensor shared_gate0, shared_up0, shared_w2_0; /* stack index 0's slice */
    int64_t shared_gate_stride, shared_up_stride, shared_w2_stride;
} RealLayer;

/* Bound to the owning RealModel by real_fill_layer on every call (not by
 * real_load), which sidesteps a return-by-value hazard: real_load builds
 * its RealModel on the stack and returns it by value, so any pointer set
 * up *inside* real_load that points back into that same local (e.g.
 * "&m.idx") would dangle once the struct is copied out on return.
 * real_fill_layer instead receives a stable `RealModel *m` (the caller's
 * settled copy) and rebinds idx/part_fds/cur_layer from it every call --
 * three redundant pointer writes per layer, correct regardless of how many
 * times the struct was copied before that. */
typedef struct RealExperts {
    const ExpertIndex *idx;
    const int *part_fds;
    int hidden, moe_inter;
    int cur_layer;

    uint8_t *gate_buf, *up_buf, *down_buf; /* reusable pread buffers, sized to idx->max_slab_bytes */
    float *g_buf, *u_buf, *h_buf;          /* [moe_inter] each */
    float *row_scratch;                    /* [max(hidden,moe_inter)], qlinear's row-dequant scratch */
} RealExperts;

typedef struct {
    Config cfg;
    Tokenizer *tok;

    void *res_base;
    size_t res_size;

    ExpertIndex idx;
    int part_fds[8];
    int64_t part_sizes[8];

    RealLayer *layers; /* [num_hidden_layers], resolved manifest lookups */
    QTensor embed, unembed;   /* token_embd.weight (Q5_K), output.weight (Q4_K) */
    const float *embed_norm, *final_norm;

    float *arena;      /* reusable per-layer f32 dequant scratch, sized to the max across all layers */
    size_t arena_floats;

    RealExperts real_exps;
} RealModel;

/* --------------------------------- real_load --------------------------------- */

static RealModel real_load(const char *config_path, const char *manifest_path,
                            const char *index_path, const char *tokenizer_path) {
    RealModel m = {0};
    m.cfg = config_load(config_path);
    m.tok = tokenizer_load(tokenizer_path);
    const Config *cfg = &m.cfg;

    /* resident.bin: convention is a sibling of the manifest, matching how
     * extract_resident.py always writes them together. */
    char *res_bin_path = dirname_join(manifest_path, "resident.bin");
    int rfd = open(res_bin_path, O_RDONLY);
    if (rfd < 0) die("open %s: %s", res_bin_path, strerror(errno));
    struct stat rst;
    if (fstat(rfd, &rst) != 0) die("fstat %s: %s", res_bin_path, strerror(errno));
    m.res_size = (size_t)rst.st_size;
    m.res_base = mmap(NULL, m.res_size, PROT_READ, MAP_PRIVATE, rfd, 0);
    if (m.res_base == MAP_FAILED) die("mmap %s: %s", res_bin_path, strerror(errno));
    close(rfd);
    free(res_bin_path);

    ResidentTable rt = manifest_load(manifest_path, m.res_base, m.res_size);

    /* GGUF parts: convention is dirname(index_path)/inkling-gguf/<part_file>,
     * matching where tools/make_index.py's default --weights-dir actually
     * lives relative to its default --out (both under weights/). */
    m.idx = index_load(index_path);
    if (m.idx.n_experts != cfg->n_routed_experts)
        die("%s: n_experts_per_layer=%d disagrees with config.json's n_routed_experts=%d",
            index_path, m.idx.n_experts, cfg->n_routed_experts);
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        if (cfg->layer_is_sparse[i] && (i >= m.idx.n_layers_alloc || !m.idx.by_layer[i].gate))
            die("%s: MoE layer %d (sparse per config.json) missing from the expert index", index_path, i);
    }

    char *gguf_dir = dirname_join(index_path, "inkling-gguf");
    for (int i = 0; i < m.idx.n_parts; i++) {
        char partpath[1024];
        int n = snprintf(partpath, sizeof partpath, "%s/%s", gguf_dir, m.idx.part_files[i]);
        if (n < 0 || (size_t)n >= sizeof partpath) die("GGUF part path too long: %s/%s", gguf_dir, m.idx.part_files[i]);
        int fd = open(partpath, O_RDONLY);
        if (fd < 0) die("open %s: %s", partpath, strerror(errno));
        struct stat pst;
        if (fstat(fd, &pst) != 0) die("fstat %s: %s", partpath, strerror(errno));
        if (fcntl(fd, F_NOCACHE, 1) != 0) die("fcntl F_NOCACHE %s: %s", partpath, strerror(errno));
        m.part_fds[i] = fd;
        m.part_sizes[i] = (int64_t)pst.st_size;
    }
    free(gguf_dir);

    /* Bounds-check every streamed expert slot against its part's actual
     * size -- the streamed-tensor analogue of manifest_load's res_size
     * check above. */
    for (int layer = 0; layer < m.idx.n_layers_alloc; layer++) {
        MoeLayerIndex *mli = &m.idx.by_layer[layer];
        if (!mli->gate) continue;
        ExpertSlot *slot_arrays[3] = {mli->gate, mli->up, mli->down};
        const char *slot_names[3] = {"gate", "up", "down"};
        for (int s = 0; s < 3; s++) {
            for (int e = 0; e < m.idx.n_experts; e++) {
                ExpertSlot *sl = &slot_arrays[s][e];
                int64_t psize = m.part_sizes[sl->part_idx];
                if (sl->abs_offset + sl->nbytes > psize)
                    die("%s: layer %d slot '%s' expert %d: [%lld,%lld) exceeds part %d size %lld bytes",
                        index_path, layer, slot_names[s], e, (long long)sl->abs_offset,
                        (long long)(sl->abs_offset + sl->nbytes), sl->part_idx, (long long)psize);
            }
        }
    }

    /* Top-level tensors: no dequant, referenced straight into res_base --
     * both are consumed row-at-a-time (qrow for embedding lookups, qlinear
     * for the final logits projection). */
    m.embed = resident_qtensor(&rt, "token_embd.weight");
    m.unembed = resident_qtensor(&rt, "output.weight");
    m.embed_norm = resident_f32(&rt, "token_embd_norm.weight");
    m.final_norm = resident_f32(&rt, "output_norm.weight");
    if (m.embed.out_dim != cfg->vocab_size || m.unembed.out_dim != cfg->vocab_size)
        die("%s: token_embd/output row counts disagree with config.json's vocab_size=%d", manifest_path, cfg->vocab_size);

    /* Per-layer resolution: strcmp-based manifest lookups, done once here
     * (not on the hot forward path). */
    m.layers = xcalloc((size_t)cfg->num_hidden_layers, sizeof(RealLayer));
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        RealLayer *rl = &m.layers[i];
        rl->attn_norm = real_layer_f32(&rt, i, "attn_norm.weight");
        rl->mlp_norm  = real_layer_f32(&rt, i, "ffn_norm.weight");
        rl->q_norm    = real_layer_f32(&rt, i, "attn_q_norm.weight");
        rl->k_norm    = real_layer_f32(&rt, i, "attn_k_norm.weight");
        rl->rel_proj  = real_layer_f32(&rt, i, "attn_rel_proj.weight");
        rl->k_sconv_w = real_layer_f32(&rt, i, "shortconv_k.weight");
        rl->v_sconv_w = real_layer_f32(&rt, i, "shortconv_v.weight");
        rl->attn_sconv_w = real_layer_f32(&rt, i, "shortconv_attn.weight");
        rl->mlp_sconv_w  = real_layer_f32(&rt, i, "shortconv_mlp.weight");

        rl->wq = real_layer_qtensor(&rt, i, "attn_q.weight");
        rl->wk = real_layer_qtensor(&rt, i, "attn_k.weight");
        rl->wv = real_layer_qtensor(&rt, i, "attn_v.weight");
        rl->wr = real_layer_qtensor(&rt, i, "attn_r.weight");
        rl->wo = real_layer_qtensor(&rt, i, "attn_output.weight");

        if (!cfg->layer_is_sparse[i]) {
            rl->dense_gate = real_layer_qtensor(&rt, i, "ffn_gate.weight");
            rl->dense_up   = real_layer_qtensor(&rt, i, "ffn_up.weight");
            rl->dense_w2   = real_layer_qtensor(&rt, i, "ffn_down.weight");
            rl->dense_global_scale = real_layer_f32(&rt, i, "ffn_gscale.weight");
        } else {
            rl->router_w = real_layer_f32(&rt, i, "ffn_gate_inp.weight");
            rl->router_bias = real_layer_f32(&rt, i, "exp_probs_b.bias");
            rl->router_global_scale = real_layer_f32(&rt, i, "ffn_gscale.weight");
            rl->shared_gate0 = real_layer_qtensor_stacked(&rt, i, "ffn_gate_shexp.weight", cfg->n_shared_experts, &rl->shared_gate_stride);
            rl->shared_up0   = real_layer_qtensor_stacked(&rt, i, "ffn_up_shexp.weight",   cfg->n_shared_experts, &rl->shared_up_stride);
            rl->shared_w2_0  = real_layer_qtensor_stacked(&rt, i, "ffn_down_shexp.weight", cfg->n_shared_experts, &rl->shared_w2_stride);
        }
    }

    /* Arena sizing: max f32 footprint across ALL layers (dense vs MoE
     * differ; sliding vs global attn dims can too) -- allocate once, reuse
     * every layer/decode-step. */
    size_t max_floats = 0;
    for (int i = 0; i < cfg->num_hidden_layers; i++) {
        RealLayer *rl = &m.layers[i];
        size_t f = 0;
        f += (size_t)rl->wq.out_dim * (size_t)rl->wq.in_dim;
        f += (size_t)rl->wk.out_dim * (size_t)rl->wk.in_dim;
        f += (size_t)rl->wv.out_dim * (size_t)rl->wv.in_dim;
        f += (size_t)rl->wr.out_dim * (size_t)rl->wr.in_dim;
        f += (size_t)rl->wo.out_dim * (size_t)rl->wo.in_dim;
        if (!cfg->layer_is_sparse[i]) {
            f += 2 * (size_t)rl->dense_gate.out_dim * (size_t)rl->dense_gate.in_dim; /* interleaved w13 */
            f += (size_t)rl->dense_w2.out_dim * (size_t)rl->dense_w2.in_dim;
        } else {
            f += (size_t)cfg->n_shared_experts * 2 * (size_t)rl->shared_gate0.out_dim * (size_t)rl->shared_gate0.in_dim;
            f += (size_t)cfg->n_shared_experts * (size_t)rl->shared_w2_0.out_dim * (size_t)rl->shared_w2_0.in_dim;
        }
        if (f > max_floats) max_floats = f;
    }
    m.arena_floats = max_floats;
    m.arena = xmalloc(sizeof(float) * m.arena_floats);

    /* RealExperts: everything except idx/part_fds/cur_layer (rebound per
     * call by real_fill_layer -- see the RealExperts comment above) can be
     * set here since these are independent heap allocations, stable
     * regardless of how many times the enclosing RealModel gets copied. */
    m.real_exps.hidden = cfg->hidden_size;
    m.real_exps.moe_inter = cfg->moe_intermediate_size;
    m.real_exps.cur_layer = -1;
    size_t slab_cap = (size_t)m.idx.max_slab_bytes;
    if (slab_cap == 0) slab_cap = 1;
    m.real_exps.gate_buf = xmalloc(slab_cap);
    m.real_exps.up_buf   = xmalloc(slab_cap);
    m.real_exps.down_buf = xmalloc(slab_cap);
    m.real_exps.g_buf = xmalloc(sizeof(float) * (size_t)cfg->moe_intermediate_size);
    m.real_exps.u_buf = xmalloc(sizeof(float) * (size_t)cfg->moe_intermediate_size);
    m.real_exps.h_buf = xmalloc(sizeof(float) * (size_t)cfg->moe_intermediate_size);
    size_t row_scratch_floats = (size_t)cfg->hidden_size;
    if ((size_t)cfg->moe_intermediate_size > row_scratch_floats) row_scratch_floats = (size_t)cfg->moe_intermediate_size;
    m.real_exps.row_scratch = xmalloc(sizeof(float) * row_scratch_floats);

    return m;
}

/* ----------------------- layer materialization + streaming ------------------ */

/* Dequantizes every row of a resident QTensor into the arena at *cursor,
 * advancing *cursor and *used; dies on arena overflow. Returns the pointer
 * to the block just written (the LayerWeights field's value). */
static const float *real_dequant_all_rows(const QTensor *w, float **cursor, size_t *used, size_t cap, const char *what) {
    size_t n = (size_t)w->out_dim * (size_t)w->in_dim;
    if (*used + n > cap) die("real: arena overflow dequantizing '%s' (%zu + %zu > %zu floats)", what, *used, n, cap);
    float *dst = *cursor;
    for (int64_t r = 0; r < w->out_dim; r++) qrow(w, r, dst + (size_t)r * (size_t)w->in_dim);
    *cursor += n;
    *used += n;
    return dst;
}

/* Builds an [n_stack, 2*out_dim, in_dim] interleaved gate/up block (row
 * 2i=gate_i, 2i+1=up_i per w13_row/model_load's confirmed on-disk
 * convention) from two separate on-disk tensors, n_stack of them stacked
 * (dense: n_stack=1, strides unused; shared experts: n_stack=2). */
static const float *real_dequant_w13_interleaved(const QTensor *gate, const QTensor *up, int n_stack,
                                                  int64_t gate_stride, int64_t up_stride,
                                                  float **cursor, size_t *used, size_t cap, const char *what) {
    int64_t out_dim = gate->out_dim, in_dim = gate->in_dim;
    size_t n = (size_t)n_stack * 2 * (size_t)out_dim * (size_t)in_dim;
    if (*used + n > cap) die("real: arena overflow dequantizing '%s' (%zu + %zu > %zu floats)", what, *used, n, cap);
    float *dst = *cursor;
    for (int s = 0; s < n_stack; s++) {
        QTensor gs = *gate; gs.data = (const uint8_t *)gate->data + (size_t)s * (size_t)gate_stride;
        QTensor us = *up;   us.data = (const uint8_t *)up->data   + (size_t)s * (size_t)up_stride;
        for (int64_t i = 0; i < out_dim; i++) {
            qrow(&gs, i, dst + ((size_t)s * 2 * (size_t)out_dim + (size_t)(2 * i)) * (size_t)in_dim);
            qrow(&us, i, dst + ((size_t)s * 2 * (size_t)out_dim + (size_t)(2 * i + 1)) * (size_t)in_dim);
        }
    }
    *cursor += n;
    *used += n;
    return dst;
}

/* Builds an [n_stack, out_dim, in_dim] concatenation (no interleave -- used
 * for shared_w2, which oracle mode stores the same way: experts stacked
 * along the outer axis, each a plain [out_dim,in_dim] matrix). */
static const float *real_dequant_stacked(const QTensor *base, int n_stack, int64_t stride,
                                          float **cursor, size_t *used, size_t cap, const char *what) {
    size_t n = (size_t)n_stack * (size_t)base->out_dim * (size_t)base->in_dim;
    if (*used + n > cap) die("real: arena overflow dequantizing '%s' (%zu + %zu > %zu floats)", what, *used, n, cap);
    float *dst = *cursor;
    for (int s = 0; s < n_stack; s++) {
        QTensor ss = *base; ss.data = (const uint8_t *)base->data + (size_t)s * (size_t)stride;
        for (int64_t r = 0; r < base->out_dim; r++)
            qrow(&ss, r, dst + ((size_t)s * (size_t)base->out_dim + (size_t)r) * (size_t)base->in_dim);
    }
    *cursor += n;
    *used += n;
    return dst;
}

/* Fills a transient LayerWeights that decoder_layer_forward can consume
 * unchanged, from one layer's resident tensors (dequantized fresh into
 * `arena` this call) plus the streamed-expert seam for sparse layers. */
static void real_fill_layer(RealModel *m, int layer, float *arena, LayerWeights *lw) {
    const Config *cfg = &m->cfg;
    const RealLayer *rl = &m->layers[layer];
    memset(lw, 0, sizeof(*lw));

    lw->is_sliding = cfg->layer_is_sliding[layer];
    lw->is_sparse  = cfg->layer_is_sparse[layer];
    lw->num_heads    = lw->is_sliding ? cfg->swa_num_attention_heads : cfg->num_attention_heads;
    lw->num_kv_heads = lw->is_sliding ? cfg->swa_num_key_value_heads : cfg->num_key_value_heads;
    lw->head_dim     = lw->is_sliding ? cfg->swa_head_dim : cfg->head_dim;
    lw->rel_extent   = lw->is_sliding ? cfg->sliding_window_size : cfg->rel_extent;
    lw->q_dim  = lw->num_heads * lw->head_dim;
    lw->kv_dim = lw->num_kv_heads * lw->head_dim;
    lw->r_dim  = lw->num_heads * cfg->d_rel;

    lw->attn_norm = rl->attn_norm;
    lw->mlp_norm  = rl->mlp_norm;
    lw->q_norm    = rl->q_norm;
    lw->k_norm    = rl->k_norm;
    lw->rel_proj  = rl->rel_proj;
    lw->k_sconv_w = rl->k_sconv_w;
    lw->v_sconv_w = rl->v_sconv_w;
    lw->attn_sconv_w = rl->attn_sconv_w;
    lw->mlp_sconv_w  = rl->mlp_sconv_w;

    float *cursor = arena;
    size_t used = 0;
    size_t cap = m->arena_floats;

    lw->wq = real_dequant_all_rows(&rl->wq, &cursor, &used, cap, "attn_q.weight");
    lw->wk = real_dequant_all_rows(&rl->wk, &cursor, &used, cap, "attn_k.weight");
    lw->wv = real_dequant_all_rows(&rl->wv, &cursor, &used, cap, "attn_v.weight");
    lw->wr = real_dequant_all_rows(&rl->wr, &cursor, &used, cap, "attn_r.weight");
    lw->wo = real_dequant_all_rows(&rl->wo, &cursor, &used, cap, "attn_output.weight");

    if (!lw->is_sparse) {
        lw->dense_w13 = real_dequant_w13_interleaved(&rl->dense_gate, &rl->dense_up, 1, 0, 0,
                                                      &cursor, &used, cap, "ffn_gate/up.weight");
        lw->dense_w2 = real_dequant_all_rows(&rl->dense_w2, &cursor, &used, cap, "ffn_down.weight");
        lw->dense_global_scale = rl->dense_global_scale;
    } else {
        lw->router_w = rl->router_w;
        lw->router_bias = rl->router_bias;
        lw->router_global_scale = rl->router_global_scale;
        lw->shared_w13 = real_dequant_w13_interleaved(&rl->shared_gate0, &rl->shared_up0, cfg->n_shared_experts,
                                                       rl->shared_gate_stride, rl->shared_up_stride,
                                                       &cursor, &used, cap, "ffn_gate/up_shexp.weight");
        lw->shared_w2 = real_dequant_stacked(&rl->shared_w2_0, cfg->n_shared_experts, rl->shared_w2_stride,
                                              &cursor, &used, cap, "ffn_down_shexp.weight");

        /* Rebind + set the current layer on the OWNER's RealExperts (see the
         * RealExperts comment above for why this can't happen in real_load). */
        m->real_exps.idx = &m->idx;
        m->real_exps.part_fds = m->part_fds;
        m->real_exps.cur_layer = layer;
        lw->real_exps = &m->real_exps;
    }
}

static void pread_exact(int fd, void *buf, size_t n, int64_t offset, const char *what) {
    ssize_t got = pread(fd, buf, n, offset);
    if (got < 0) die("pread %s: %s", what, strerror(errno));
    if ((size_t)got != n) die("pread %s: short read (%zd of %zu bytes)", what, got, n);
}

/* Full implementation of the seam declared near LayerWeights and stubbed in
 * Step 1. Streams one selected routed expert's gate/up/down slabs (index
 * sidecar gives their exact byte ranges), wraps each as a QTensor, and
 * computes silu(gate)*up -> down_proj with qlinear -- the EXACT same
 * silu/gating expression as mlp_moe_forward's in-memory branch (the
 * per-expert weight is applied by the caller, identically for both paths). */
static void real_expert_ffn(const struct RealExperts *re, int expert_idx, const float *x, float *expert_out) {
    if (re->cur_layer < 0 || re->cur_layer >= re->idx->n_layers_alloc)
        die("real: expert_ffn called with no current layer bound (cur_layer=%d)", re->cur_layer);
    const MoeLayerIndex *mli = &re->idx->by_layer[re->cur_layer];
    if (!mli->gate) die("real: MoE layer %d has no expert index entry", re->cur_layer);
    if (expert_idx < 0 || expert_idx >= re->idx->n_experts)
        die("real: expert %d out of range [0,%d)", expert_idx, re->idx->n_experts);

    const ExpertSlot *ge = &mli->gate[expert_idx];
    const ExpertSlot *ue = &mli->up[expert_idx];
    const ExpertSlot *de = &mli->down[expert_idx];

    pread_exact(re->part_fds[ge->part_idx], re->gate_buf, (size_t)ge->nbytes, ge->abs_offset, "expert gate slab");
    pread_exact(re->part_fds[ue->part_idx], re->up_buf,   (size_t)ue->nbytes, ue->abs_offset, "expert up slab");
    pread_exact(re->part_fds[de->part_idx], re->down_buf, (size_t)de->nbytes, de->abs_offset, "expert down slab");

    QTensor gate_qt = { ge->ggml_type, re->gate_buf, re->moe_inter, re->hidden };
    QTensor up_qt   = { ue->ggml_type, re->up_buf,   re->moe_inter, re->hidden };
    QTensor down_qt = { de->ggml_type, re->down_buf, re->hidden,    re->moe_inter };

    qlinear(&gate_qt, x, re->g_buf, re->row_scratch);
    qlinear(&up_qt,   x, re->u_buf, re->row_scratch);
    for (int i = 0; i < re->moe_inter; i++) re->h_buf[i] = silu_f(re->g_buf[i]) * re->u_buf[i];
    qlinear(&down_qt, re->h_buf, expert_out, re->row_scratch);
}

/* Cache sized directly from Config (real mode has no permanent LayerWeights
 * array to read kv_dim off of -- LayerWeights is transient/rebuilt every
 * layer/call by real_fill_layer). Mirrors cache_create's per-layer kv_dim
 * formula exactly (model_load's is_sliding ? swa_* : *). */
static Cache *cache_create_cfg(const Config *cfg, int cap) {
    Cache *c = xmalloc(sizeof(Cache));
    c->cap = cap;
    c->num_layers = cfg->num_hidden_layers;
    c->layers = xcalloc((size_t)c->num_layers, sizeof(LayerCache));
    int Km1 = cfg->conv_kernel_size - 1;
    int hidden = cfg->hidden_size;
    for (int i = 0; i < c->num_layers; i++) {
        LayerCache *lc = &c->layers[i];
        int is_sliding = cfg->layer_is_sliding[i];
        int kv_dim = is_sliding ? cfg->swa_num_key_value_heads * cfg->swa_head_dim
                                : cfg->num_key_value_heads * cfg->head_dim;
        lc->k = xcalloc((size_t)cap * (size_t)kv_dim, sizeof(float));
        lc->v = xcalloc((size_t)cap * (size_t)kv_dim, sizeof(float));
        lc->k_hist = xcalloc((size_t)Km1 * (size_t)kv_dim, sizeof(float));
        lc->v_hist = xcalloc((size_t)Km1 * (size_t)kv_dim, sizeof(float));
        lc->attn_hist = xcalloc((size_t)Km1 * (size_t)hidden, sizeof(float));
        lc->mlp_hist = xcalloc((size_t)Km1 * (size_t)hidden, sizeof(float));
    }
    return c;
}

/* Mirrors model_forward_chunk's sequence exactly (embed -> layers ->
 * final norm -> mup divide -> logits), with real_fill_layer supplying each
 * layer's transient LayerWeights instead of a permanently-loaded one. */
static void real_forward_chunk(RealModel *m, Cache *cache, const int *token_ids, int T, int start_pos,
                                float *logits_out /* [T,unpadded] or NULL */) {
    const Config *cfg = &m->cfg;
    int hidden = cfg->hidden_size;

    float *x = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    float *erow = xmalloc(sizeof(float) * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        qrow(&m->embed, token_ids[t], erow);
        rmsnorm(erow, m->embed_norm, hidden, (float)cfg->rms_norm_eps, x + (size_t)t * hidden);
    }
    free(erow);

    LayerWeights lw;
    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        real_fill_layer(m, l, m->arena, &lw);
        decoder_layer_forward(cfg, &lw, &cache->layers[l], x, T, start_pos);
    }

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++)
        rmsnorm(x + (size_t)t * hidden, m->final_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
    free(x);

    if (logits_out) {
        int unpadded = cfg->unpadded_vocab_size;
        float mup = (float)cfg->logits_mup_width_multiplier;
        QTensor unembed_unpadded = m->unembed;
        unembed_unpadded.out_dim = unpadded; /* only the unpadded rows (architecture-notes.md sec.8) */
        float *h = xmalloc(sizeof(float) * (size_t)hidden);
        float *row_scratch = xmalloc(sizeof(float) * (size_t)hidden);
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < hidden; d++) h[d] = normed[(size_t)t * hidden + d] / mup;
            qlinear(&unembed_unpadded, h, logits_out + (size_t)t * unpadded, row_scratch);
        }
        free(h);
        free(row_scratch);
    }
    free(normed);
}

static double elapsed_ms(struct timespec t0, struct timespec t1) {
    return (double)(t1.tv_sec - t0.tv_sec) * 1000.0 + (double)(t1.tv_nsec - t0.tv_nsec) / 1e6;
}

#define REAL_MAX_PROMPT_IDS 8192

/* Tokenizes (no BOS prepend -- add_bos_token=false), prefills, then greedily
 * decodes, printing each token's id, decoded text, and the wall time of the
 * forward pass that produced the logits it was picked from (prefill time for
 * the first token, else the previous decode step's). Stops at n_gen or eos. */
static void real_generate(RealModel *m, const char *prompt, int n_gen) {
    const Config *cfg = &m->cfg;
    int unpadded = cfg->unpadded_vocab_size;

    int32_t *ids = xmalloc(sizeof(int32_t) * (size_t)REAL_MAX_PROMPT_IDS);
    int n_prompt = tokenizer_encode(m->tok, prompt, ids, REAL_MAX_PROMPT_IDS);
    int *ids32 = xmalloc(sizeof(int) * (size_t)n_prompt);
    for (int i = 0; i < n_prompt; i++) ids32[i] = (int)ids[i];

    Cache *cache = cache_create_cfg(cfg, n_prompt + n_gen + 8);

    float *logits = xmalloc(sizeof(float) * (size_t)n_prompt * (size_t)unpadded);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    real_forward_chunk(m, cache, ids32, n_prompt, 0, logits);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = elapsed_ms(t0, t1);
    free(ids32);

    float *cur = xmalloc(sizeof(float) * (size_t)unpadded);
    memcpy(cur, logits + (size_t)(n_prompt - 1) * unpadded, sizeof(float) * (size_t)unpadded);
    free(logits);

    int eos_id = tokenizer_eos_id(m->tok);
    int pos = n_prompt;
    char textbuf[256];
    for (int i = 0; i < n_gen; i++) {
        int tok = argmax_f(cur, unpadded);
        tokenizer_decode(m->tok, &tok, 1, textbuf, sizeof textbuf);
        printf("%d\t%s\t%.1f ms\n", tok, textbuf, ms);
        fflush(stdout);
        if (tok == eos_id) break;
        int next_id = tok;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        real_forward_chunk(m, cache, &next_id, 1, pos, cur);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms = elapsed_ms(t0, t1);
        pos++;
    }
    cache_free(cache);
    free(ids);
    free(cur);
}

/* --logits-only: prefill the prompt, print the top-10 first-decode-step
 * logits and exit (informational only -- Tasks 15-16 validate correctness). */
static void real_print_top_logits(RealModel *m, const char *prompt) {
    const Config *cfg = &m->cfg;
    int unpadded = cfg->unpadded_vocab_size;

    int32_t *ids = xmalloc(sizeof(int32_t) * (size_t)REAL_MAX_PROMPT_IDS);
    int n_prompt = tokenizer_encode(m->tok, prompt, ids, REAL_MAX_PROMPT_IDS);
    int *ids32 = xmalloc(sizeof(int) * (size_t)n_prompt);
    for (int i = 0; i < n_prompt; i++) ids32[i] = (int)ids[i];

    Cache *cache = cache_create_cfg(cfg, n_prompt);
    float *logits = xmalloc(sizeof(float) * (size_t)n_prompt * (size_t)unpadded);
    real_forward_chunk(m, cache, ids32, n_prompt, 0, logits);
    const float *last = logits + (size_t)(n_prompt - 1) * unpadded;

    int idx[10];
    float val[10];
    for (int k = 0; k < 10; k++) { idx[k] = -1; val[k] = -INFINITY; }
    for (int v = 0; v < unpadded; v++) {
        float x = last[v];
        if (x > val[9]) {
            int p = 9;
            while (p > 0 && x > val[p - 1]) { val[p] = val[p - 1]; idx[p] = idx[p - 1]; p--; }
            val[p] = x;
            idx[p] = v;
        }
    }
    printf("top-10 first-token logits for a %d-token prompt:\n", n_prompt);
    for (int k = 0; k < 10; k++) printf("  [%d] id=%d logit=%.6f\n", k, idx[k], (double)val[k]);

    free(logits);
    free(ids32);
    free(ids);
    cache_free(cache);
}

/* ------------------------------ --smoke mode -------------------------------- */
/* Synthetic, weights-free end-to-end plumbing check (Step 4): loads a
 * tools/make_smoke_fixture.py-generated manifest/index through the SAME
 * manifest_load/index_load/resident_qtensor/qlinear code paths as --real,
 * preads each synthetic expert slab and byte-compares it against
 * smoke_expected.json's recorded first-8-bytes, then runs the engine's
 * actual qlinear() on a synthetic resident Q8_0 tensor and compares against
 * an expected vector, bitwise. */
static void run_smoke(const char *dir) {
    char idxpath[1024];
    int n = snprintf(idxpath, sizeof idxpath, "%s/index.json", dir);
    if (n < 0 || (size_t)n >= sizeof idxpath) die("smoke: path too long: %s", dir);
    ExpertIndex idx = index_load(idxpath);

    char resbin_path[1024];
    n = snprintf(resbin_path, sizeof resbin_path, "%s/resident.bin", dir);
    if (n < 0 || (size_t)n >= sizeof resbin_path) die("smoke: path too long: %s", dir);
    int rfd = open(resbin_path, O_RDONLY);
    if (rfd < 0) die("open %s: %s", resbin_path, strerror(errno));
    struct stat rst;
    if (fstat(rfd, &rst) != 0) die("fstat %s: %s", resbin_path, strerror(errno));
    size_t res_size = (size_t)rst.st_size;
    void *res_base = mmap(NULL, res_size, PROT_READ, MAP_PRIVATE, rfd, 0);
    if (res_base == MAP_FAILED) die("mmap %s: %s", resbin_path, strerror(errno));
    close(rfd);

    char manifest_path[1024];
    n = snprintf(manifest_path, sizeof manifest_path, "%s/resident-manifest.json", dir);
    if (n < 0 || (size_t)n >= sizeof manifest_path) die("smoke: path too long: %s", dir);
    ResidentTable rt = manifest_load(manifest_path, res_base, res_size);

    /* Same dirname(index_path)+"/inkling-gguf" convention real_load uses. */
    char *gguf_dir = dirname_join(idxpath, "inkling-gguf");
    int part_fds[8];
    int64_t part_sizes[8];
    for (int i = 0; i < idx.n_parts; i++) {
        char partpath[1024];
        int pn = snprintf(partpath, sizeof partpath, "%s/%s", gguf_dir, idx.part_files[i]);
        if (pn < 0 || (size_t)pn >= sizeof partpath) die("smoke: GGUF part path too long");
        int fd = open(partpath, O_RDONLY);
        if (fd < 0) die("open %s: %s", partpath, strerror(errno));
        struct stat pst;
        if (fstat(fd, &pst) != 0) die("fstat %s: %s", partpath, strerror(errno));
        part_fds[i] = fd;
        part_sizes[i] = (int64_t)pst.st_size;
    }
    free(gguf_dir);

    char exppath[1024];
    n = snprintf(exppath, sizeof exppath, "%s/smoke_expected.json", dir);
    if (n < 0 || (size_t)n >= sizeof exppath) die("smoke: path too long: %s", dir);
    size_t elen;
    char *ebuf = read_file(exppath, &elen);
    JsonValue *eroot = json_parse(ebuf, elen);

    int layer = json_req_int(eroot, "layer", exppath);
    if (layer < 0 || layer >= idx.n_layers_alloc || !idx.by_layer[layer].gate)
        die("smoke: expected layer %d not present in the index", layer);
    MoeLayerIndex *mli = &idx.by_layer[layer];

    const JsonValue *experts = json_get(eroot, "experts");
    if (!experts || experts->type != JSON_ARR) die("smoke: expected.json missing 'experts'");
    size_t buf_cap = (size_t)idx.max_slab_bytes > 0 ? (size_t)idx.max_slab_bytes : 1;
    uint8_t *buf = xmalloc(buf_cap);
    for (size_t ei = 0; ei < experts->arr_count; ei++) {
        const JsonValue *eobj = experts->arr_items[ei];
        int e = json_req_int(eobj, "expert", exppath);
        const char *slot_names[3] = {"gate", "up", "down"};
        ExpertSlot *slot_arrays[3] = {mli->gate, mli->up, mli->down};
        for (int k = 0; k < 3; k++) {
            const JsonValue *slot = json_get(eobj, slot_names[k]);
            if (!slot) die("smoke: expected.json expert %d missing slot '%s'", e, slot_names[k]);
            int64_t exp_offset = json_req_i64(slot, "abs_offset", exppath);
            int64_t exp_nbytes = json_req_i64(slot, "nbytes", exppath);
            const char *first8_hex = json_req_str(slot, "first8_hex", exppath);
            if (strlen(first8_hex) != 16) die("smoke: first8_hex must be 16 hex chars");
            ExpertSlot *sl = &slot_arrays[k][e];
            if (sl->abs_offset != exp_offset || sl->nbytes != exp_nbytes)
                die("smoke: expert %d slot '%s': parsed index (offset=%lld,nbytes=%lld) != expected.json (offset=%lld,nbytes=%lld)",
                    e, slot_names[k], (long long)sl->abs_offset, (long long)sl->nbytes,
                    (long long)exp_offset, (long long)exp_nbytes);
            if (sl->abs_offset + sl->nbytes > part_sizes[sl->part_idx])
                die("smoke: expert %d slot '%s': slab exceeds part size", e, slot_names[k]);
            pread_exact(part_fds[sl->part_idx], buf, (size_t)sl->nbytes, sl->abs_offset, slot_names[k]);
            uint8_t want[8];
            for (int b = 0; b < 8; b++) {
                unsigned int byteval;
                if (sscanf(first8_hex + 2 * b, "%2x", &byteval) != 1) die("smoke: bad hex in first8_hex");
                want[b] = (uint8_t)byteval;
            }
            if (memcmp(buf, want, 8) != 0) die("smoke: expert %d slot '%s': first 8 bytes mismatch", e, slot_names[k]);
        }
    }
    free(buf);

    const JsonValue *resident = json_get(eroot, "resident");
    if (resident && resident->type == JSON_ARR) {
        for (size_t i = 0; i < resident->arr_count; i++) {
            const JsonValue *r = resident->arr_items[i];
            const char *rname = json_req_str(r, "name", exppath);
            int64_t roff = json_req_i64(r, "offset", exppath);
            int64_t rnbytes = json_req_i64(r, "nbytes", exppath);
            const ResidentEntry *e = resident_get(&rt, rname);
            if (e->offset != roff || e->nbytes != rnbytes)
                die("smoke: resident tensor '%s' offset/nbytes mismatch vs expected.json", rname);
        }
    }

    const JsonValue *qlin = json_get(eroot, "qlinear");
    if (!qlin) die("smoke: expected.json missing 'qlinear'");
    const char *qname = json_req_str(qlin, "tensor_name", exppath);
    QTensor q = resident_qtensor(&rt, qname);
    const JsonValue *xarr = json_get(qlin, "x");
    const JsonValue *yarr = json_get(qlin, "y");
    if (!xarr || xarr->type != JSON_ARR || (int64_t)xarr->arr_count != q.in_dim)
        die("smoke: qlinear.x length mismatch");
    if (!yarr || yarr->type != JSON_ARR || (int64_t)yarr->arr_count != q.out_dim)
        die("smoke: qlinear.y length mismatch");
    float *xv = xmalloc(sizeof(float) * (size_t)q.in_dim);
    for (int64_t i = 0; i < q.in_dim; i++) xv[i] = (float)json_num(xarr->arr_items[i], 0);
    float *yv = xmalloc(sizeof(float) * (size_t)q.out_dim);
    float *row_scratch = xmalloc(sizeof(float) * (size_t)q.in_dim);
    qlinear(&q, xv, yv, row_scratch);
    for (int64_t v = 0; v < q.out_dim; v++) {
        float expect = (float)json_num(yarr->arr_items[v], 0);
        if (yv[v] != expect)
            die("smoke: qlinear mismatch at row %lld: got %.9g expected %.9g", (long long)v, (double)yv[v], (double)expect);
    }
    free(xv);
    free(yv);
    free(row_scratch);

    printf("smoke ok\n");
}

/* ================================== main =================================== */

int main(int argc, char **argv) {
    const char *dump_path = NULL;
    const char *smoke_dir = NULL;
    const char *prompt = "The capital of France is";
    int n_gen = 32;
    int do_real = 0, do_mlock = 0, logits_only = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump-acts") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else if (strcmp(argv[i], "--real") == 0) {
            do_real = 1;
        } else if (strcmp(argv[i], "--smoke") == 0 && i + 1 < argc) {
            smoke_dir = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--n-gen") == 0 && i + 1 < argc) {
            n_gen = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mlock") == 0) {
            do_mlock = 1;
        } else if (strcmp(argv[i], "--logits-only") == 0) {
            logits_only = 1;
        } else {
            fprintf(stderr,
                "usage: %s [--dump-acts FILE]\n"
                "       %s --smoke DIR\n"
                "       %s --real [--prompt TEXT] [--n-gen N] [--mlock] [--logits-only]\n"
                "\n"
                "--real paths default to docs/inkling-config.json, weights/resident-manifest.json,\n"
                "weights/inkling-ud-q2_k_xl.sepia-index.json, weights/tokenizer.bin -- override with\n"
                "the SEPIA_REAL_CONFIG_PATH / SEPIA_REAL_MANIFEST_PATH / SEPIA_REAL_INDEX_PATH /\n"
                "SEPIA_REAL_TOKENIZER_PATH env vars.\n"
                "--mlock only makes sense once `extract_resident.py --verify` has confirmed\n"
                "resident.bin's bytes match its manifest -- not enforceable here, the loader trusts its caller.\n",
                argv[0], argv[0], argv[0]);
            return 2;
        }
    }

    if (smoke_dir) {
        run_smoke(smoke_dir);
        return 0;
    }

    if (do_real) {
        const char *real_config_path = env_or("SEPIA_REAL_CONFIG_PATH", "docs/inkling-config.json");
        const char *manifest_path = env_or("SEPIA_REAL_MANIFEST_PATH", "weights/resident-manifest.json");
        const char *index_path = env_or("SEPIA_REAL_INDEX_PATH", "weights/inkling-ud-q2_k_xl.sepia-index.json");
        const char *tokenizer_path = env_or("SEPIA_REAL_TOKENIZER_PATH", "weights/tokenizer.bin");

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        RealModel rm = real_load(real_config_path, manifest_path, index_path, tokenizer_path);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "real: loaded in %.2fs (resident.bin %.2f GB mmap'd, %d GGUF part(s) open)\n",
                elapsed_ms(t0, t1) / 1000.0, (double)rm.res_size / 1e9, rm.idx.n_parts);

        if (do_mlock) {
            if (mlock(rm.res_base, rm.res_size) != 0) die("mlock: %s", strerror(errno));
            fprintf(stderr, "real: mlock'd resident.bin (%zu bytes)\n", rm.res_size);
        }

        if (logits_only) {
            real_print_top_logits(&rm, prompt);
            return 0;
        }

        real_generate(&rm, prompt, n_gen);
        return 0;
    }

    /* SEPIA_{CONFIG,WEIGHTS,REF}_PATH override the committed toy fixture
     * paths; unset by default, so the plain `./sepia` self-test is
     * unaffected. */
    const char *config_path = env_or("SEPIA_CONFIG_PATH", CONFIG_PATH);
    const char *weights_path = env_or("SEPIA_WEIGHTS_PATH", WEIGHTS_PATH);
    const char *ref_path = env_or("SEPIA_REF_PATH", REF_PATH);

    Model m = model_load(config_path, weights_path);

    if (dump_path) {
        OracleRef ref = load_oracle_ref(ref_path);
        int full_len = ref.full_ids.len;
        int unpadded = m.cfg.unpadded_vocab_size;
        Cache *c = cache_create(&m, full_len);
        ActDump dump = dump_open(dump_path);
        float *logits = xmalloc(sizeof(float) * (size_t)full_len * (size_t)unpadded);
        model_forward_chunk(&m, c, ref.full_ids.ids, full_len, 0, logits, &dump);
        dump_close(&dump);
        free(logits);
        cache_free(c);
        printf("wrote activation dump to %s\n", dump_path);
        return 0;
    }

    int ok = run_self_test(&m, ref_path);
    return ok ? 0 : 1;
}

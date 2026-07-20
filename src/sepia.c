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
#include <pthread.h>
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

/* --gpu-compare-tiny's capture lists (below) grow by doubling; realloc(3)
 * itself tolerates a NULL pointer (equivalent to malloc), matching xmalloc's
 * out-of-memory die() policy. */
static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory (realloc %zu bytes)", n);
    return q;
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
#include "sepia_gpu.h"
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

static void qlinear(const QTensor *w, const float *x, float *y, float *row_scratch) {
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

/* ============================ op capture (GPU compare) ==================== */
/* --gpu-compare-tiny (local-only): while g_opcap is set, the call sites
 * below additionally snapshot their real (input, params, CPU-computed
 * output) tuples into the per-op-kind lists here, alongside the existing,
 * completely unmodified math -- these are pure side-effect captures, never
 * read back by the CPU forward pass itself, so with g_opcap==0 (every other
 * mode: the self-test, --real, --dump-acts, --smoke) every function in this
 * file computes byte-identical results to before this section existed. The
 * harness (run_gpu_compare_tiny, near main()) drives one T=32 prefill plus
 * one incremental T=1 decode step (continuing the same cache, so its sconv
 * instances see nonzero history -- the prefill alone only ever sees the
 * cache's zero-initialized history) through the UNMODIFIED
 * model_forward_chunk with g_opcap set, then replays each captured instance
 * through the matching sepia_gpu_* kernel and reports max-relative-error per
 * op kind.
 *
 * g_opcap_selected_layer restricts the per-layer op kinds (matvec, sconv,
 * softmax, silu_mul, and the per-layer rmsnorm sites) to layer 0 and the
 * last layer, set by model_forward_chunk's layer loop just before each
 * decoder_layer_forward call -- layer 0 is dense-MLP + sliding-window
 * attention and the last layer is sparse-MoE + global attention in every
 * config this codebase has seen (tiny oracle and the real checkpoint alike,
 * sec.1/sec.2), so this one pair naturally exercises both MLP branches and
 * both attention-window branches without a special case for either. The
 * always-on sites (embed_norm, final_norm rmsnorm in model_forward_chunk)
 * aren't layer-scoped and use g_opcap directly. */
static int g_opcap = 0;
static int g_opcap_selected_layer = 0;

static float *fdup(const float *src, size_t n) {
    float *d = xmalloc(sizeof(float) * (n ? n : 1));
    memcpy(d, src, sizeof(float) * n);
    return d;
}

static void opcap_ensure(void **items, int *count, int *cap, size_t elem_size) {
    if (*count == *cap) {
        *cap = *cap ? *cap * 2 : 64;
        *items = xrealloc(*items, elem_size * (size_t)(*cap));
    }
}

typedef struct { int n; float eps; float *x, *w, *y; } CapRmsnorm;
typedef struct { CapRmsnorm *items; int count, cap; } CapRmsnormList;
static CapRmsnormList g_cap_rmsnorm;

static void cap_push_rmsnorm(const float *x, const float *w, int n, float eps, const float *y) {
    opcap_ensure((void **)&g_cap_rmsnorm.items, &g_cap_rmsnorm.count, &g_cap_rmsnorm.cap, sizeof(CapRmsnorm));
    CapRmsnorm *it = &g_cap_rmsnorm.items[g_cap_rmsnorm.count++];
    it->n = n;
    it->eps = eps;
    it->x = fdup(x, (size_t)n);
    it->w = fdup(w, (size_t)n);
    it->y = fdup(y, (size_t)n);
}

typedef struct { int out_dim, in_dim; float *w, *x, *y; } CapMatvec;
typedef struct { CapMatvec *items; int count, cap; } CapMatvecList;
static CapMatvecList g_cap_matvec;

/* w must be [out_dim,in_dim] row-major CONTIGUOUS (row stride ==
 * in_dim*sizeof(float)) -- sepia_gpu_matvec's Task 3 scope, see its header
 * doc. Callers with an interleaved/strided source (dense_w13/experts_w13's
 * gate/up interleave) are out of scope here; only contiguous-row matvec call
 * sites (wq/wk/wv/wr/wo, dense_w2, experts_w2/shared_w2, router_w) push. */
static void cap_push_matvec(const float *w, const float *x, int out_dim, int in_dim, const float *y) {
    opcap_ensure((void **)&g_cap_matvec.items, &g_cap_matvec.count, &g_cap_matvec.cap, sizeof(CapMatvec));
    CapMatvec *it = &g_cap_matvec.items[g_cap_matvec.count++];
    it->out_dim = out_dim;
    it->in_dim = in_dim;
    it->w = fdup(w, (size_t)out_dim * (size_t)in_dim);
    it->x = fdup(x, (size_t)in_dim);
    it->y = fdup(y, (size_t)out_dim);
}

typedef struct { int n; float *g, *u, *y; } CapSiluMul;
typedef struct { CapSiluMul *items; int count, cap; } CapSiluMulList;
static CapSiluMulList g_cap_silu;

static void cap_push_silu_mul(const float *g, const float *u, int n, const float *y) {
    opcap_ensure((void **)&g_cap_silu.items, &g_cap_silu.count, &g_cap_silu.cap, sizeof(CapSiluMul));
    CapSiluMul *it = &g_cap_silu.items[g_cap_silu.count++];
    it->n = n;
    it->g = fdup(g, (size_t)n);
    it->u = fdup(u, (size_t)n);
    it->y = fdup(y, (size_t)n);
}

typedef struct { int n; float *a, *b, *y; } CapAdd;
typedef struct { CapAdd *items; int count, cap; } CapAddList;
static CapAddList g_cap_add;

static void cap_push_add(const float *a, const float *b, int n, const float *y) {
    opcap_ensure((void **)&g_cap_add.items, &g_cap_add.count, &g_cap_add.cap, sizeof(CapAdd));
    CapAdd *it = &g_cap_add.items[g_cap_add.count++];
    it->n = n;
    it->a = fdup(a, (size_t)n);
    it->b = fdup(b, (size_t)n);
    it->y = fdup(y, (size_t)n);
}

/* scores: raw pre-softmax scores (content + relative-position bias, before
 * the max-subtract/exp/normalize steps); y: the final normalized attention
 * weights for that same (t,h) -- i.e. exactly what attn_forward_chunk's
 * inlined stable-softmax computes internally but never materializes as an
 * array (it folds the normalize step into the V-weighted-sum loop instead).
 * The capture call site reconstructs y explicitly for comparison purposes
 * only; it does not change what attn_forward_chunk computes or returns. */
typedef struct { int n; float *scores, *y; } CapSoftmax;
typedef struct { CapSoftmax *items; int count, cap; } CapSoftmaxList;
static CapSoftmaxList g_cap_softmax;

static void cap_push_softmax(const float *scores, int n, const float *y) {
    opcap_ensure((void **)&g_cap_softmax.items, &g_cap_softmax.count, &g_cap_softmax.cap, sizeof(CapSoftmax));
    CapSoftmax *it = &g_cap_softmax.items[g_cap_softmax.count++];
    it->n = n;
    it->scores = fdup(scores, (size_t)n);
    it->y = fdup(y, (size_t)n);
}

/* hist: the [K-1,C] history window as sconv_apply saw it BEFORE the call
 * (sconv_apply mutates it in place to prepare the next chunk's history --
 * that update is Task 8's concern, not replayed or compared here, see
 * sepia_gpu_sconv's header doc). */
typedef struct { int C, K, T; float *w, *hist, *in, *y; } CapSconv;
typedef struct { CapSconv *items; int count, cap; } CapSconvList;
static CapSconvList g_cap_sconv;

static void cap_push_sconv(const float *w, const float *hist, const float *in, int C, int K, int T, const float *y) {
    opcap_ensure((void **)&g_cap_sconv.items, &g_cap_sconv.count, &g_cap_sconv.cap, sizeof(CapSconv));
    CapSconv *it = &g_cap_sconv.items[g_cap_sconv.count++];
    it->C = C;
    it->K = K;
    it->T = T;
    it->w = fdup(w, (size_t)C * (size_t)K);
    it->hist = fdup(hist, (size_t)(K - 1) * (size_t)C);
    it->in = fdup(in, (size_t)T * (size_t)C);
    it->y = fdup(y, (size_t)T * (size_t)C);
}

/* Task 7 Gate B (--gpu-compare-tiny's attention-swap mode): one instance per
 * attn_forward_chunk call (restricted to layer 0 / last layer by
 * g_opcap_selected_layer, same as every other Cap* site). Captures exactly
 * what's needed to replay the WHOLE attention block (projections -> sconv ->
 * per-head norm -> cache assembly -> rel-project -> banded attention -> wo)
 * via GPU dispatches and compare against attn_forward_chunk's own real
 * output: the pre-call cache history (k_pre/v_pre -- the cache slice BEFORE
 * this call's writes -- and k_hist_pre/v_hist_pre -- the sconv history
 * before sconv_apply mutates it in place, mirroring CapSconv's own "hist
 * before" convention), the block's input (x_normed) and final output (out).
 * `cfg`/`lw` are stored as plain pointers (both outlive the harness's
 * comparison pass -- they point into the still-live Model this call came
 * from), not duplicated: only the per-call activation/cache tensors need
 * their own copies since those are mutated or freed as the forward pass
 * continues. x_normed/out are copied here (fdup) since the caller keeps
 * using/owning those buffers afterward; k_pre/v_pre/k_hist_pre/v_hist_pre
 * arrive as throwaway snapshots the caller made solely for this call, so
 * cap_push_attn takes ownership of those allocations directly instead of
 * copying them a second time. */
typedef struct {
    const Config *cfg;
    const LayerWeights *lw;
    int T, start_pos;
    float *x_normed;   /* [T,hidden] */
    float *k_pre;      /* [start_pos,kv_dim] (start_pos may be 0) */
    float *v_pre;      /* [start_pos,kv_dim] */
    float *k_hist_pre; /* [K-1,kv_dim] */
    float *v_hist_pre; /* [K-1,kv_dim] */
    float *out;        /* [T,hidden] -- attn_forward_chunk's real (post-wo) output */
} CapAttn;
typedef struct { CapAttn *items; int count, cap; } CapAttnList;
static CapAttnList g_cap_attn;

static void cap_push_attn(const Config *cfg, const LayerWeights *lw, int T, int start_pos,
                           const float *x_normed, float *k_pre, float *v_pre,
                           float *k_hist_pre, float *v_hist_pre, const float *out) {
    opcap_ensure((void **)&g_cap_attn.items, &g_cap_attn.count, &g_cap_attn.cap, sizeof(CapAttn));
    CapAttn *it = &g_cap_attn.items[g_cap_attn.count++];
    int hidden = cfg->hidden_size;
    it->cfg = cfg;
    it->lw = lw;
    it->T = T;
    it->start_pos = start_pos;
    it->x_normed = fdup(x_normed, (size_t)T * (size_t)hidden);
    /* k_pre/v_pre/k_hist_pre/v_hist_pre arrive already fdup'd by the caller
     * (gpu_cap_k_pre etc. in attn_forward_chunk, made solely to survive to
     * this call) -- take ownership of those allocations directly instead of
     * fdup'ing them a second time; the caller no longer frees them. */
    it->k_pre = k_pre;
    it->v_pre = v_pre;
    it->k_hist_pre = k_hist_pre;
    it->v_hist_pre = v_hist_pre;
    it->out = fdup(out, (size_t)T * (size_t)hidden);
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

    /* Task 7 Gate B capture only: snapshot the cache slice BEFORE this
     * call's writes (the memcpy below at start_pos..start_pos+T overwrites
     * lc->k/lc->v, so this must happen before that point) -- see CapAttn's
     * doc comment, above. Pure side-effect capture; ownership passes to
     * cap_push_attn near the end of this function (it stores these
     * allocations directly, no re-copy), so nothing here is freed by this
     * function. No effect on the math below when g_opcap_selected_layer is
     * 0. */
    float *gpu_cap_k_pre = NULL, *gpu_cap_v_pre = NULL;
    float *gpu_cap_k_hist_pre = NULL, *gpu_cap_v_hist_pre = NULL;
    if (g_opcap_selected_layer) {
        gpu_cap_k_pre = fdup(lc->k, (size_t)start_pos * (size_t)kv_dim);
        gpu_cap_v_pre = fdup(lc->v, (size_t)start_pos * (size_t)kv_dim);
    }

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
        if (g_opcap_selected_layer) {
            cap_push_matvec(lw->wq, xt, q_dim, hidden, q_raw + (size_t)t * q_dim);
            cap_push_matvec(lw->wk, xt, kv_dim, hidden, k_raw + (size_t)t * kv_dim);
            cap_push_matvec(lw->wv, xt, kv_dim, hidden, v_raw + (size_t)t * kv_dim);
            cap_push_matvec(lw->wr, xt, r_dim, hidden, r_raw + (size_t)t * r_dim);
        }
    }

    float *k_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    if (g_opcap_selected_layer) {
        /* Task 7 Gate B: these survive to cap_push_attn near the end of this
         * function, which takes ownership of them directly (alongside
         * gpu_cap_k_pre/v_pre above) rather than freeing/re-copying them --
         * everything else about this branch is unchanged from before Task 7. */
        gpu_cap_k_hist_pre = fdup(lc->k_hist, (size_t)(K - 1) * (size_t)kv_dim);
        gpu_cap_v_hist_pre = fdup(lc->v_hist, (size_t)(K - 1) * (size_t)kv_dim);
        sconv_apply(lw->k_sconv_w, kv_dim, K, lc->k_hist, k_raw, T, k_sconv);
        sconv_apply(lw->v_sconv_w, kv_dim, K, lc->v_hist, v_raw, T, v_sconv);
        cap_push_sconv(lw->k_sconv_w, gpu_cap_k_hist_pre, k_raw, kv_dim, K, T, k_sconv);
        cap_push_sconv(lw->v_sconv_w, gpu_cap_v_hist_pre, v_raw, kv_dim, K, T, v_sconv);
    } else {
        sconv_apply(lw->k_sconv_w, kv_dim, K, lc->k_hist, k_raw, T, k_sconv);
        sconv_apply(lw->v_sconv_w, kv_dim, K, lc->v_hist, v_raw, T, v_sconv);
    }
    free(k_raw);
    free(v_raw);

    /* per-head RMSNorm: q_norm/k_norm applied after the view into
     * [...,heads,head_dim] but before transpose -- order only matters for
     * which axis is reduced (always head_dim), so a flat per-head-slice
     * normalize is exactly equivalent (sec.2). No norm on v. */
    for (int t = 0; t < T; t++) {
        for (int h = 0; h < H; h++) {
            float *qh = q_raw + (size_t)t * q_dim + (size_t)h * Dh;
            if (g_opcap_selected_layer) {
                float *qh_before = fdup(qh, (size_t)Dh);
                rmsnorm(qh, lw->q_norm, Dh, (float)cfg->rms_norm_eps, qh);
                cap_push_rmsnorm(qh_before, lw->q_norm, Dh, (float)cfg->rms_norm_eps, qh);
                free(qh_before);
            } else {
                rmsnorm(qh, lw->q_norm, Dh, (float)cfg->rms_norm_eps, qh);
            }
        }
        for (int h = 0; h < Hkv; h++) {
            float *kh = k_sconv + (size_t)t * kv_dim + (size_t)h * Dh;
            if (g_opcap_selected_layer) {
                float *kh_before = fdup(kh, (size_t)Dh);
                rmsnorm(kh, lw->k_norm, Dh, (float)cfg->rms_norm_eps, kh);
                cap_push_rmsnorm(kh_before, lw->k_norm, Dh, (float)cfg->rms_norm_eps, kh);
                free(kh_before);
            } else {
                rmsnorm(kh, lw->k_norm, Dh, (float)cfg->rms_norm_eps, kh);
            }
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
            float *raw_scores_before = g_opcap_selected_layer ? fdup(scores, (size_t)n_kv) : NULL;
            double sum_exp = 0.0;
            for (int i = 0; i < n_kv; i++) {
                double e = exp((double)(scores[i] - max_score));
                scores[i] = (float)e;
                sum_exp += e;
            }
            if (raw_scores_before) {
                float *weights = xmalloc(sizeof(float) * (size_t)n_kv);
                for (int i = 0; i < n_kv; i++) weights[i] = (float)((double)scores[i] / sum_exp);
                cap_push_softmax(raw_scores_before, n_kv, weights);
                free(weights);
                free(raw_scores_before);
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

    for (int t = 0; t < T; t++) {
        linear(lw->wo, hidden, q_dim, attn_concat + (size_t)t * q_dim, out + (size_t)t * hidden);
        if (g_opcap_selected_layer)
            cap_push_matvec(lw->wo, attn_concat + (size_t)t * q_dim, hidden, q_dim, out + (size_t)t * hidden);
    }
    free(attn_concat);

    if (g_opcap_selected_layer) {
        /* cap_push_attn takes ownership of gpu_cap_k_pre/v_pre/k_hist_pre/
         * v_hist_pre (stores the allocations directly rather than
         * re-copying them) -- they must NOT be freed here; they stay live
         * inside the CapAttn record for Gate B's later replay pass. */
        cap_push_attn(cfg, lw, T, start_pos, x_normed, gpu_cap_k_pre, gpu_cap_v_pre,
                      gpu_cap_k_hist_pre, gpu_cap_v_hist_pre, out);
    }
}

/* ================================== MLP ================================= */
/* Dense SwiGLU FFN (layers < dense_mlp_idx). global_scale multiplies the
 * FFN *output* here (sec.9) -- contrast with the MoE router below, where
 * the analogous per-layer scale multiplies the routing *weights* instead. */
static void mlp_dense_forward(const LayerWeights *lw, int hidden, int dense_inter, const float *x, float *out) {
    float *h = xmalloc(sizeof(float) * (size_t)dense_inter);
    float *g_vec = g_opcap_selected_layer ? xmalloc(sizeof(float) * (size_t)dense_inter) : NULL;
    float *u_vec = g_opcap_selected_layer ? xmalloc(sizeof(float) * (size_t)dense_inter) : NULL;
    int two_inter = 2 * dense_inter;
    for (int i = 0; i < dense_inter; i++) {
        float g = dotf(w13_row(lw->dense_w13, 0, two_inter, hidden, 0, i), x, hidden);
        float u = dotf(w13_row(lw->dense_w13, 0, two_inter, hidden, 1, i), x, hidden);
        h[i] = silu_f(g) * u;
        if (g_opcap_selected_layer) {
            g_vec[i] = g;
            u_vec[i] = u;
        }
    }
    if (g_opcap_selected_layer) {
        cap_push_silu_mul(g_vec, u_vec, dense_inter, h);
        free(g_vec);
        free(u_vec);
    }
    float gscale = lw->dense_global_scale[0];
    for (int d = 0; d < hidden; d++) {
        const float *row = lw->dense_w2 + (size_t)d * dense_inter;
        out[d] = dotf(row, h, dense_inter) * gscale;
    }
    if (g_opcap_selected_layer) {
        /* NOTE: out[] carries an extra *gscale the matvec kernel itself
         * doesn't apply -- captured for the matvec op's dot-product step
         * only, so the "expected" y here is the pre-gscale value (recomputed
         * via a plain dot, gscale divided back out from the already-scaled
         * out[] would just reintroduce float rounding for no reason). */
        float *pre_scale = xmalloc(sizeof(float) * (size_t)hidden);
        for (int d = 0; d < hidden; d++) pre_scale[d] = dotf(lw->dense_w2 + (size_t)d * dense_inter, h, dense_inter);
        cap_push_matvec(lw->dense_w2, h, hidden, dense_inter, pre_scale);
        free(pre_scale);
    }
    free(h);
}

/* Router selection + mixing-weight computation (sec.6): given raw router
 * logits (pre-sigmoid, pre-bias) for n_routed routed experts plus n_shared
 * always-on shared "sink" experts, fills topk_idx[0..topk) with the selected
 * routed expert indices and weights[0..topk+n_shared) with the final mixing
 * weight for each (routed slots first, then shared slots) -- the aux-loss-
 * free bias (router_bias) selects experts (topk) but is NOT part of the
 * mixing weight, which comes from log_sigmoid/logsumexp over the RAW logits
 * of the selected routed experts plus the shared-expert-sink logits.
 * Extracted out of mlp_moe_forward (below) so the GPU forward path (Task 8)
 * can reuse this EXACT, already-verified selection/weight math on GPU-
 * readback router logits instead of reimplementing it -- see
 * .superpowers/sdd/task-8-report.md's "MoE readback design" note. Caller-
 * allocated: topk_idx[topk], weights[topk+n_shared]. */
static void moe_route_select(const Config *cfg, const LayerWeights *lw, const float *router_logits,
                              int *topk_idx, float *weights) {
    int n_routed = cfg->n_routed_experts, n_shared = cfg->n_shared_experts, topk = cfg->num_experts_per_tok;

    float *scores_for_choice = xmalloc(sizeof(float) * (size_t)n_routed);
    for (int i = 0; i < n_routed; i++) scores_for_choice[i] = sigmoid_f(router_logits[i]) + lw->router_bias[i];

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
    free(scores_for_choice);

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

    double gs = (double)lw->router_global_scale[0];
    for (int j = 0; j < n_sel; j++) {
        double w = exp((double)log_probs[j] - lse);
        w *= cfg->route_scale * gs; /* norm_after_topk's renormalization, then route_scale*global_scale (sec.6) */
        weights[j] = (float)w;
    }

    free(topk_logits);
    free(log_probs);
}

static void mlp_moe_forward(const Config *cfg, const LayerWeights *lw, const float *x, float *out) {
    int hidden = cfg->hidden_size;
    int n_routed = cfg->n_routed_experts, n_shared = cfg->n_shared_experts, topk = cfg->num_experts_per_tok;
    int n_total = n_routed + n_shared;
    int moe_inter = cfg->moe_intermediate_size;
    int two_inter = 2 * moe_inter;

    float *router_logits = xmalloc(sizeof(float) * (size_t)n_total);
    for (int i = 0; i < n_total; i++) router_logits[i] = dotf(lw->router_w + (size_t)i * hidden, x, hidden);
    if (g_opcap_selected_layer) cap_push_matvec(lw->router_w, x, n_total, hidden, router_logits);

    int *topk_idx = xmalloc(sizeof(int) * (size_t)topk);
    int n_sel = topk + n_shared;
    float *weights = xmalloc(sizeof(float) * (size_t)n_sel);
    moe_route_select(cfg, lw, router_logits, topk_idx, weights);

    for (int d = 0; d < hidden; d++) out[d] = 0.0f;

    float *h = xmalloc(sizeof(float) * (size_t)moe_inter);
    float *expert_out = xmalloc(sizeof(float) * (size_t)hidden);
    float *g_vec = g_opcap_selected_layer ? xmalloc(sizeof(float) * (size_t)moe_inter) : NULL;
    float *u_vec = g_opcap_selected_layer ? xmalloc(sizeof(float) * (size_t)moe_inter) : NULL;
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
                if (g_opcap_selected_layer) {
                    g_vec[i] = g;
                    u_vec[i] = u;
                }
            }
            if (g_opcap_selected_layer) cap_push_silu_mul(g_vec, u_vec, moe_inter, h);
            for (int d = 0; d < hidden; d++) {
                const float *row = lw->experts_w2 + ((size_t)e * hidden + (size_t)d) * moe_inter;
                expert_out[d] = dotf(row, h, moe_inter);
            }
            if (g_opcap_selected_layer)
                cap_push_matvec(lw->experts_w2 + (size_t)e * (size_t)hidden * (size_t)moe_inter, h,
                                 hidden, moe_inter, expert_out);
        }
        float wj = weights[j];
        for (int d = 0; d < hidden; d++) out[d] += expert_out[d] * wj; /* routed: weight applied AFTER down_proj (sec.7) */
    }
    free(expert_out);
    free(g_vec);
    free(u_vec);

    float *shared_out = g_opcap_selected_layer ? xmalloc(sizeof(float) * (size_t)hidden) : NULL;
    for (int s = 0; s < n_shared; s++) {
        float gamma = weights[topk + s];
        /* No cap_push_silu_mul here (intentional, unlike the routed branch
         * above): h[i] below folds *gamma into the activation itself, so it
         * isn't the same computation sepia_silu_mul_f32 implements -- see
         * task-3-report.md's "Comparison-harness op scope" note. */
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
            float v = dotf(row, h, moe_inter);
            if (g_opcap_selected_layer) shared_out[d] = v;
            out[d] += v;
        }
        /* shared_w2 is the same contiguous row-major [hidden,moe_inter]
         * family as experts_w2 above (both are plain down-projections),
         * just previously uncaptured -- shared_out holds the pure Wx result
         * (before accumulating into out) so the capture matches what
         * sepia_gpu_matvec actually computes. */
        if (g_opcap_selected_layer)
            cap_push_matvec(lw->shared_w2 + (size_t)s * (size_t)hidden * (size_t)moe_inter, h,
                             hidden, moe_inter, shared_out);
    }
    free(shared_out);

    free(h);
    free(weights);
    free(topk_idx);
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
    for (int t = 0; t < T; t++) {
        rmsnorm(x + (size_t)t * hidden, lw->attn_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
        if (g_opcap_selected_layer)
            cap_push_rmsnorm(x + (size_t)t * hidden, lw->attn_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
    }

    float *attn_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    attn_forward_chunk(cfg, lw, lc, normed, T, start_pos, attn_out);

    float *attn_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    if (g_opcap_selected_layer) {
        float *attn_hist_before = fdup(lc->attn_hist, (size_t)(K - 1) * (size_t)hidden);
        sconv_apply(lw->attn_sconv_w, hidden, K, lc->attn_hist, attn_out, T, attn_sconv_out);
        cap_push_sconv(lw->attn_sconv_w, attn_hist_before, attn_out, hidden, K, T, attn_sconv_out);
        free(attn_hist_before);
        float *x_before = fdup(x, (size_t)T * (size_t)hidden);
        for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += attn_sconv_out[i];
        cap_push_add(x_before, attn_sconv_out, T * hidden, x);
        free(x_before);
    } else {
        sconv_apply(lw->attn_sconv_w, hidden, K, lc->attn_hist, attn_out, T, attn_sconv_out);
        for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += attn_sconv_out[i];
    }
    free(attn_out);
    free(attn_sconv_out);

    for (int t = 0; t < T; t++) {
        rmsnorm(x + (size_t)t * hidden, lw->mlp_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
        if (g_opcap_selected_layer)
            cap_push_rmsnorm(x + (size_t)t * hidden, lw->mlp_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
    }

    float *mlp_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        if (lw->is_sparse)
            mlp_moe_forward(cfg, lw, normed + (size_t)t * hidden, mlp_out + (size_t)t * hidden);
        else
            mlp_dense_forward(lw, hidden, cfg->dense_intermediate_size, normed + (size_t)t * hidden, mlp_out + (size_t)t * hidden);
    }
    free(normed);

    float *mlp_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    if (g_opcap_selected_layer) {
        float *mlp_hist_before = fdup(lc->mlp_hist, (size_t)(K - 1) * (size_t)hidden);
        sconv_apply(lw->mlp_sconv_w, hidden, K, lc->mlp_hist, mlp_out, T, mlp_sconv_out);
        cap_push_sconv(lw->mlp_sconv_w, mlp_hist_before, mlp_out, hidden, K, T, mlp_sconv_out);
        free(mlp_hist_before);
        float *x_before = fdup(x, (size_t)T * (size_t)hidden);
        for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += mlp_sconv_out[i];
        cap_push_add(x_before, mlp_sconv_out, T * hidden, x);
        free(x_before);
    } else {
        sconv_apply(lw->mlp_sconv_w, hidden, K, lc->mlp_hist, mlp_out, T, mlp_sconv_out);
        for (size_t i = 0; i < (size_t)T * hidden; i++) x[i] += mlp_sconv_out[i];
    }
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
        if (g_opcap) cap_push_rmsnorm(erow, m->embed_norm, hidden, (float)cfg->rms_norm_eps, x + (size_t)t * hidden);
    }
    if (dump) dump_capture(dump, "embed_out", x, T, hidden);

    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        /* --gpu-compare-tiny only: restrict the per-layer op captures (Cap*
         * lists other than rmsnorm's two always-on sites above/below) to
         * layer 0 and the last layer -- see the "op capture" section's
         * doc comment (near cache_free) for why that pair is sufficient. */
        g_opcap_selected_layer = g_opcap && (l == 0 || l == cfg->num_hidden_layers - 1);
        decoder_layer_forward(cfg, &m->layers[l], &cache->layers[l], x, T, start_pos);
        if (dump) {
            char name[32];
            snprintf(name, sizeof name, "layer%d.out", l);
            dump_capture(dump, name, x, T, hidden);
        }
    }
    g_opcap_selected_layer = 0;

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        rmsnorm(x + (size_t)t * hidden, m->final_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
        if (g_opcap) cap_push_rmsnorm(x + (size_t)t * hidden, m->final_norm, hidden, (float)cfg->rms_norm_eps, normed + (size_t)t * hidden);
    }
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

/* Task 8: GPU-mode twin of model_forward_chunk above -- identical per-layer
 * math (attn_forward_chunk's / mlp_dense_forward's / mlp_moe_forward's own
 * formulas, including moe_route_select's routing decision), computed via
 * sepia_gpu_* dispatches instead of linear/dotf/rmsnorm. Defined near the
 * other --gpu-compare-tiny/--gpu-compare-attn machinery below (it needs
 * gpu_upload_f32 and the small dispatch-wrapper helpers introduced there);
 * forward-declared here so run_self_test (which selects between the two
 * paths on --metal) can call it before that point in the file. logits_out
 * mirrors model_forward_chunk's own NULL-to-skip convention; there is no
 * `dump` parameter -- --dump-acts stays CPU-path only (never combined with
 * --metal in main()). */
static void model_forward_chunk_gpu(const Model *m, Cache *cache, const int *token_ids, int T, int start_pos,
                                     float *logits_out /*[T,unpadded_vocab] or NULL*/);

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
 * separately, per the brief. Returns 1 iff both hit a perfect score.
 *
 * use_gpu (Task 8): routes every model_forward_chunk call through the GPU
 * twin (model_forward_chunk_gpu) instead -- same Cache, same oracle ref,
 * same prefill-then-decode shape; this is the ONLY thing --metal changes
 * about the self-test (`./sepia` with no flag never sets this, so the plain
 * CPU self-test stays byte-identical). */
static int run_self_test(const Model *m, const char *ref_path, int use_gpu) {
    OracleRef ref = load_oracle_ref(ref_path);
    int unpadded = m->cfg.unpadded_vocab_size;
    int full_len = ref.full_ids.len;
    int prompt_len = ref.prompt_ids.len;

    /* --- prefill / teacher forcing --- */
    Cache *tf_cache = cache_create(m, full_len);
    float *logits = xmalloc(sizeof(float) * (size_t)full_len * (size_t)unpadded);
    if (use_gpu)
        model_forward_chunk_gpu(m, tf_cache, ref.full_ids.ids, full_len, 0, logits);
    else
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
    if (use_gpu)
        model_forward_chunk_gpu(m, dc, ref.prompt_ids.ids, prompt_len, 0, plogits);
    else
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
        if (use_gpu)
            model_forward_chunk_gpu(m, dc, &next_id, 1, cur_pos, cur_logits);
        else
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

/* Task 9: the GPU-resident twin of RealLayer -- same per-layer tensors, but
 * every weight is (ggml_type, BYTE OFFSET into the one wrapped gpu_res_buf,
 * dims) instead of an arena/mmap pointer. Built once at real_load time (only
 * when --metal initialized the GPU runtime first) directly from the
 * already-resolved, already-bounds-checked RealLayer fields -- see
 * gpu_off_of/gpu_qtensor_off/real_gpu_layer_build below. F32 resident
 * tensors that the GPU path only ever needs as scalar host reads (dense/
 * router global_scale, router_bias) stay accessed via RealLayer's existing
 * host pointers; only tensors an actual GPU kernel dispatch reads get an
 * offset here. */
typedef struct { int ggml_type; size_t w_off; int64_t out_dim, in_dim; } GpuQTensor;

typedef struct {
    size_t attn_norm_off, mlp_norm_off, q_norm_off, k_norm_off, rel_proj_off;
    size_t k_sconv_off, v_sconv_off, attn_sconv_off, mlp_sconv_off;
    GpuQTensor wq, wk, wv, wr, wo;

    /* dense only */
    GpuQTensor dense_gate, dense_up, dense_w2;

    /* sparse only */
    size_t router_w_off; /* F32 [n_total,hidden] */
    GpuQTensor shared_gate0, shared_up0, shared_w2_0; /* stack index 0's slice; caller offsets by
                                                        * s*stride the same way RealLayer's do */
    int64_t shared_gate_stride, shared_up_stride, shared_w2_stride;
} RealLayerGpu;

/* Task 9: persistent per-layer GPU KV cache + sconv-history buffers (Shared
 * storage, so both GPU kernels and the one host readback the attention
 * block needs -- see real_attn_forward_chunk_gpu's tau-sync -- can touch
 * them). Allocated once per generation (gpu_cache_create_cfg), zeroed at
 * creation (sconv history and a fresh KV cache both start at all-zero,
 * matching cache_create_cfg's xcalloc exactly) and mutated in place by
 * GPU-side dispatches only (sepia_gpu_copy for cache writes,
 * sepia_gpu_sconv_hist_roll for history) -- never re-uploaded wholesale the
 * way the tiny GPU path (Task 8) re-uploads its full k/v history every
 * call, which is the actual point of keeping these persistent: attention
 * reads directly from the SAME buffer across every layer/token this
 * generation runs. */
typedef struct {
    SepiaGpuBuf *k, *v;                /* [cap, kv_dim] */
    SepiaGpuBuf *k_hist, *v_hist;      /* [Km1, kv_dim] */
    SepiaGpuBuf *attn_hist, *mlp_hist; /* [Km1, hidden] */
    int64_t kv_dim;
    int len;
} GpuLayerCache;

typedef struct {
    GpuLayerCache *layers; /* [num_hidden_layers] */
    int num_layers, cap;
} GpuCache;

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

/* ------------------------ GPU expert store (Task 10) ------------------------ */
/* LRU-streamed, mlocked GPU-resident cache for routed-expert weights --
 * replaces real_expert_ffn's per-call CPU pread+qlinear on the --metal path
 * (real_expert_ffn itself is untouched and still serves CPU real mode).
 * Ported from ds4's slab/slot/free-list/per-slot-mlock/bindless-address-
 * table shape (ds4_metal.m:8274-8410) with ds4's hotness-decayed LFU
 * replaced by plain LRU (P2 scope; ds4's fancier policy is P3 per the
 * plan). The plan's own Task 10 text calls out an "or reuse the
 * offset-addressed matvec_q against the slot's SepiaGpuBuf" alternative to
 * a new address-based kernel variant -- this implementation takes exactly
 * that path: `table[layer*n_experts+expert]` is the per-layer hit/miss
 * lookup (-1 = miss, else a slot index), and a hit's actual GPU dispatch
 * reuses the SAME offset-addressed sepia_gpu_matvec_q the resident path
 * already calls (SepiaGpuBuf* + byte offset) rather than true in-kernel
 * bindless pointer chasing. This needs zero new Metal surface -- no new
 * .metal kernels, no sepia_gpu.h/sepia_metal.m changes at all, every
 * dispatch below is an existing call. */
#define SEPIA_MOE_MAX_TOPK 16      /* compile-time bound on num_experts_per_tok;
                                    * this model's config uses 6 (SEPIA_MOE_MAX_TOPK
                                    * gives headroom, matching the SEPIA_ATTN_MAX_DH/
                                    * SEPIA_SCONV_MAX_KM1 bound-check precedent) */
#define SEPIA_EXPERT_SLAB_BYTES ((size_t)4ULL * 1024 * 1024 * 1024) /* ~4GiB, ds4-matched */

typedef struct {
    int layer, expert;        /* current owner; -1/-1 if never installed */
    int ggml_type_gate, ggml_type_up, ggml_type_down;
    int slab_idx;              /* which slab (slabs[slab_idx]) this slot lives in */
    size_t base_off;           /* gate region's byte offset within that slab;
                                 * up = base_off+region_bytes, down = base_off+2*region_bytes
                                 * (the 3 regions are laid out contiguously per slot, so ONE
                                 * mlock call covers the whole slot) */
    int mlocked;                /* mlock'd exactly once, on this slot's first install ever --
                                  * later reinstalls (different expert, same slot) reuse the
                                  * same locked VA range, no re-mlock needed */
    uint64_t safe_gen;          /* not evictable while gen_completed < safe_gen -- see
                                  * expert_cache_evict_or_get_free's header comment */
    int lru_prev, lru_next;    /* -1 = list end */
} ExpertCacheSlot;

typedef struct {
    SepiaGpuBuf **slabs;
    int n_slabs;
    size_t region_bytes;        /* per gate/up/down region: page_round(idx->max_slab_bytes) */
    size_t slot_bytes;          /* 3 * region_bytes */
    int slots_per_slab;
    int n_slots;
    ExpertCacheSlot *slots;      /* [n_slots] */
    int lru_head, lru_tail;      /* MRU / LRU list ends (index into slots[]) */

    int *table;                  /* [n_layers_alloc*n_experts], -1 = miss, else slot index */
    int n_layers_alloc, n_experts;

    /* In-flight generation stamps (ds4_metal.m:834-912's pattern, scoped down
     * to a single counter pair): gen_completed is bumped by real_gpu_end
     * every time the real-model GPU path's sepia_gpu_end() succeeds. A
     * slot's safe_gen records the generation its NEXT read will complete
     * under (gen_completed+1 at the time of access) -- eviction skips any
     * slot with safe_gen > gen_completed. Task 10 only stamped safe_gen on
     * INSTALL (a miss); Task 11 review fix: expert_cache_get now stamps it
     * on EVERY access, hit or miss. Under Task 10's fully-synchronous
     * design a hit-only gap was inert (nothing was ever actually
     * concurrent, so a hit's slot could never be evicted before its own
     * dispatch completed regardless of the stamp). Task 11 introduces real
     * overlap -- a slot that was a HIT this token can still be referenced
     * by an in-flight GPU dispatch (or a still-in-flight loader-thread
     * prefetch) when a LATER expert (same or next token) wants to evict it
     * for a miss -- so every access, not just installs, must extend the
     * slot's protection through at least the generation its own dispatch
     * will complete under. See expert_cache_get and real_mlp_moe_forward_
     * gpu's Task 11 comments for why a single "+1" (rather than tracking
     * per-access generations) stays correct: expert_cache_get is only ever
     * called again (possibly evicting this slot) after the CURRENT sparse
     * layer's own real_gpu_end() (SYNC #2) has already run, so gen_completed
     * has already advanced past whatever generation this access's stamp
     * named. */
    uint64_t gen_completed;

    /* mlock failure degrades the cache (stop trying to pin further slots,
     * one warning) rather than aborting -- Global Constraints' mlock policy.
     * Already-locked slots keep their lock; unlocked slots still work
     * correctly, just eligible for swap under real memory pressure. */
    int mlock_disabled, mlock_warned;

    uint64_t hits, misses;
    int verbose;                 /* --verbose-cache */

    /* Task 11 A/B: 0 (default) = the part_fds were opened with F_NOCACHE
     * (direct I/O, bypassing the page cache -- SEPIA's own measured 13.3GB/s
     * and ~2x-buffered result, docs/ssd-bench.md); 1 = the fds are plain
     * buffered fds and expert_cache_get fires an F_RDADVISE readahead hint
     * per region before handing the read to the loader pool (ds4's
     * alternative). Set once at expert_store_init from --expert-io-mode;
     * read-only afterward. */
    int io_pagecache;
} ExpertGpuStore;

/* Doubly-linked-list splice: detach slot `idx` from wherever it sits in the
 * LRU list. Safe to call on an already-detached node only if lru_head/tail
 * are updated consistently by the caller -- every call site below always
 * re-attaches immediately via push_head, so the list is never left with a
 * dangling member. */
static void expert_cache_lru_unlink(ExpertGpuStore *st, int idx) {
    ExpertCacheSlot *sl = &st->slots[idx];
    if (sl->lru_prev >= 0) st->slots[sl->lru_prev].lru_next = sl->lru_next;
    else st->lru_head = sl->lru_next;
    if (sl->lru_next >= 0) st->slots[sl->lru_next].lru_prev = sl->lru_prev;
    else st->lru_tail = sl->lru_prev;
    sl->lru_prev = sl->lru_next = -1;
}

static void expert_cache_lru_push_head(ExpertGpuStore *st, int idx) {
    ExpertCacheSlot *sl = &st->slots[idx];
    sl->lru_prev = -1;
    sl->lru_next = st->lru_head;
    if (st->lru_head >= 0) st->slots[st->lru_head].lru_prev = idx;
    st->lru_head = idx;
    if (st->lru_tail < 0) st->lru_tail = idx;
}

/* Picks a slot to (re)use for a cache miss: the true LRU tail, unless it (or
 * slots ahead of it) are still in-flight (safe_gen > gen_completed), in
 * which case walks toward the head looking for one that's safe. Bounded to
 * n_slots steps -- never loops forever -- and falls back to the tail slot
 * outright if every slot in the list is (implausibly) unsafe. That fallback
 * stays correct under Task 11's real overlap too, not just Task 10's
 * synchronous design: expert_cache_get is only ever called from within ONE
 * sparse layer's expert-resolution loop at a time (never two layers/tokens
 * interleaved on the host), and that loop always closes with its own
 * real_gpu_end() (SYNC #2) before the next one can start -- so at most
 * `topk` (<= SEPIA_MOE_MAX_TOPK, and n_slots >= topk is enforced at init)
 * slots can ever be simultaneously unsafe, well short of "every slot in the
 * list". Gate 4's "first mlock/eviction failure must not hang or crash"
 * concern is satisfied by this bound plus expert_slot_mlock_once's own
 * degrade-not-abort policy below. */
static int expert_cache_evict_or_get_free(ExpertGpuStore *st) {
    int idx = st->lru_tail;
    for (int steps = 0; steps < st->n_slots && idx >= 0; steps++) {
        ExpertCacheSlot *sl = &st->slots[idx];
        if (sl->layer < 0 || sl->safe_gen <= st->gen_completed) return idx;
        idx = sl->lru_prev;
    }
    return st->lru_tail;
}

/* Per-slot mlock, exactly once over that slot's lifetime (subsequent
 * reinstalls into the same physical slot for a different expert don't need
 * to re-lock -- the VA range is already pinned). On failure: warn ONCE,
 * disable further mlock attempts (the cache keeps working correctly
 * without pinning, just loses the "stays resident under memory pressure"
 * guarantee for slots locked after the failure) -- never abort. This is
 * also the answer to Gate 4's "what if the very first mlock ever attempted
 * fails": mlock_warned/mlock_disabled start at 0, the first failure sets
 * both and returns, the slot is simply left unmlocked and the cache
 * continues -- no loop, no crash. */
static void expert_slot_mlock_once(ExpertGpuStore *st, ExpertCacheSlot *sl, void *slab_base) {
    if (sl->mlocked || st->mlock_disabled) return;
    void *region = (uint8_t *)slab_base + sl->base_off;
    if (mlock(region, st->slot_bytes) != 0) {
        if (!st->mlock_warned) {
            fprintf(stderr,
                    "sepia: metal: expert-cache: mlock failed (%s) -- degrading: "
                    "further slots will not be pinned (already-pinned slots keep their lock); "
                    "the cache remains correct, just swappable under memory pressure\n",
                    strerror(errno));
            st->mlock_warned = 1;
        }
        st->mlock_disabled = 1;
        return;
    }
    sl->mlocked = 1;
}

/* Allocates slab(s) sized to fit budget_bytes (capped at the distinct
 * (layer,expert) pair count -- more slots than that can ever be used is
 * wasted GPU memory) and builds the free/LRU list + lookup table. idx's
 * OWN pointer (not stored) is only read for its VALUE fields here --
 * expert_cache_get (below, after pread_exact) takes idx/part_fds as
 * explicit per-call arguments from the caller's stable RealModel*, the
 * same rebind-per-call discipline RealExperts documents above, since
 * real_load's own `m.idx`/`m.part_fds` addresses do not survive its
 * return-by-value. */
static void expert_store_init(ExpertGpuStore *st, const ExpertIndex *idx, size_t budget_bytes,
                               int num_experts_per_tok, int verbose, int io_pagecache) {
    memset(st, 0, sizeof(*st));
    st->verbose = verbose;
    st->io_pagecache = io_pagecache;
    st->n_layers_alloc = idx->n_layers_alloc;
    st->n_experts = idx->n_experts;

    if (st->n_layers_alloc <= 0 || st->n_experts <= 0) {
        fprintf(stderr, "sepia: metal: expert-cache: no MoE layers in the index -- cache disabled\n");
        return;
    }

    size_t page = (size_t)getpagesize();
    size_t max_slab = (size_t)idx->max_slab_bytes;
    if (max_slab == 0) max_slab = 1;
    st->region_bytes = (max_slab + page - 1) / page * page;
    st->slot_bytes = st->region_bytes * 3;

    st->slots_per_slab = (int)(SEPIA_EXPERT_SLAB_BYTES / st->slot_bytes);
    if (st->slots_per_slab < 1) st->slots_per_slab = 1;

    int64_t total_pairs = (int64_t)st->n_layers_alloc * (int64_t)st->n_experts;
    int64_t want_slots = (int64_t)(budget_bytes / st->slot_bytes);
    if (want_slots > total_pairs) want_slots = total_pairs;
    if (want_slots < 1) want_slots = 1;
    st->n_slots = (int)want_slots;

    /* Task 10 review fix: a per-token expert loop installs up to topk
     * distinct experts before any sync (real_mlp_moe_forward_gpu), and
     * expert_cache_evict_or_get_free's bounded fallback assumes at least
     * that many physical slots exist -- otherwise it can hand back a slot
     * already claimed by an earlier iteration of the SAME token's loop,
     * silently overwriting one expert's weights with another's mid-token.
     * A too-small --expert-cache-gb (or a malformed value that parses to
     * ~0 via atof()) must fail loudly at load, not corrupt output later. */
    if (st->n_slots < num_experts_per_tok)
        die("metal: expert cache too small (%d slots, need at least %d for top-%d routing) -- "
            "increase --expert-cache-gb",
            st->n_slots, num_experts_per_tok, num_experts_per_tok);

    st->n_slabs = (st->n_slots + st->slots_per_slab - 1) / st->slots_per_slab;
    if (st->n_slabs < 1) st->n_slabs = 1;

    st->slabs = xcalloc((size_t)st->n_slabs, sizeof(SepiaGpuBuf *));
    for (int i = 0; i < st->n_slabs; i++) {
        int slots_this_slab = (i == st->n_slabs - 1) ? st->n_slots - i * st->slots_per_slab : st->slots_per_slab;
        size_t this_slab_bytes = (size_t)slots_this_slab * st->slot_bytes;
        st->slabs[i] = sepia_gpu_alloc(this_slab_bytes, /*gpu_private=*/0);
        if (!st->slabs[i])
            die("metal: expert-cache: failed to allocate slab %d/%d (%.2f GB)",
                i, st->n_slabs, (double)this_slab_bytes / 1e9);
    }

    st->slots = xcalloc((size_t)st->n_slots, sizeof(ExpertCacheSlot));
    for (int i = 0; i < st->n_slots; i++) {
        ExpertCacheSlot *sl = &st->slots[i];
        sl->layer = -1;
        sl->expert = -1;
        sl->slab_idx = i / st->slots_per_slab;
        sl->base_off = (size_t)(i % st->slots_per_slab) * st->slot_bytes;
        sl->lru_prev = i - 1;
        sl->lru_next = (i == st->n_slots - 1) ? -1 : i + 1;
    }
    st->lru_head = 0;
    st->lru_tail = st->n_slots - 1;

    size_t table_n = (size_t)st->n_layers_alloc * (size_t)st->n_experts;
    st->table = xmalloc(sizeof(int) * table_n);
    for (size_t i = 0; i < table_n; i++) st->table[i] = -1;

    fprintf(stderr,
            "sepia: metal: expert-cache: %d slots across %d slab(s) (%.2f MB/slot) -- "
            "%.2f GB budget covers %.1f%% of %lld distinct (layer,expert) pairs\n",
            st->n_slots, st->n_slabs, (double)st->slot_bytes / 1e6, (double)budget_bytes / 1e9,
            100.0 * (double)st->n_slots / (double)total_pairs, (long long)total_pairs);
}

static void expert_store_shutdown(ExpertGpuStore *st) {
    if (st->slabs) {
        for (int i = 0; i < st->n_slabs; i++) sepia_gpu_free(st->slabs[i]);
        free(st->slabs);
    }
    free(st->slots);
    free(st->table);
    memset(st, 0, sizeof(*st));
}

/* Forward declaration: the Task 11 loader thread pool is defined further
 * down (near expert_cache_get, its main consumer), but real_load (below)
 * starts it right after expert_store_init -- see the pool's own definition
 * for the full design. */
static void expert_loader_pool_start(void);

typedef struct {
    Config cfg;
    Tokenizer *tok;

    void *res_base;
    size_t res_size;

    /* Zero-copy Metal wrap of res_base/res_size (SepiaGpuBuf*, opaque),
     * set only when --metal initialized the GPU runtime before real_load
     * ran. void* rather than SepiaGpuBuf* keeps this struct's field types
     * plain C regardless of what sepia_gpu.h's opaque type expands to;
     * cast back to SepiaGpuBuf* at the sepia_gpu_* call sites. No
     * functional use yet -- later P2 tasks read weights through it. */
    void *gpu_res_buf;

    ExpertIndex idx;
    int part_fds[8];
    int64_t part_sizes[8];

    RealLayer *layers; /* [num_hidden_layers], resolved manifest lookups */
    QTensor embed, unembed;   /* token_embd.weight (Q5_K), output.weight (Q4_K) */
    const float *embed_norm, *final_norm;

    float *arena;      /* reusable per-layer f32 dequant scratch, sized to the max across all layers */
    size_t arena_floats;

    RealExperts real_exps;

    /* Task 9: non-NULL only when gpu_res_buf is set (--metal). gpu_layers
     * mirrors `layers` field-for-field but with GPU offsets instead of
     * pointers; gpu_unembed is output.weight's GPU descriptor (the logits
     * matvec_q). */
    RealLayerGpu *gpu_layers; /* [num_hidden_layers] */
    GpuQTensor gpu_unembed;

    /* Task 10: the GPU-resident LRU expert cache (embedded by value, safe
     * across real_load's return-by-value the same way real_exps is --
     * ExpertGpuStore's own pointer fields are all stable heap allocations,
     * never back-pointers into `m` itself). Zeroed (all-miss, empty) until
     * expert_store_init runs inside real_load's existing GPU-init block. */
    ExpertGpuStore expert_store;
} RealModel;

/* Computes the byte offset of a resident-buffer pointer (already validated
 * once by manifest_load's own [offset,offset+nbytes) <= res_size check at
 * load time) relative to m->res_base, for use as an `_off` GPU op's weight-
 * offset argument into the SAME wrapped gpu_res_buf. Task 4's review-tracked
 * hardening carried to Task 9: bounds-check offset+nbytes against the
 * buffer length AGAIN here (independent of manifest_load's check -- this is
 * the last point before a GPU kernel dispatch would silently read out of
 * bounds) and assert the offset is EVEN (every quant block's f16 scale
 * fields, and this model's own half-precision reads, need 2-byte
 * alignment; an odd offset would silently corrupt them) -- die() loudly at
 * load time here, never at dispatch. */
static size_t gpu_off_of(const RealModel *m, const void *ptr, size_t nbytes, const char *what) {
    const uint8_t *base = (const uint8_t *)m->res_base;
    const uint8_t *p = (const uint8_t *)ptr;
    if (!p || p < base || p > base + m->res_size)
        die("metal: resident tensor '%s' pointer is outside resident.bin's mapped range", what);
    size_t off = (size_t)(p - base);
    if (off % 2 != 0)
        die("metal: resident tensor '%s' has an odd byte offset %zu into resident.bin "
            "(half-precision reads require 2-byte alignment)", what, off);
    if (off + nbytes > m->res_size)
        die("metal: resident tensor '%s' [%zu,%zu) exceeds resident.bin size %zu bytes",
            what, off, off + nbytes, m->res_size);
    return off;
}

/* Same bounds+alignment discipline as gpu_off_of, for a QTensor (quantized
 * weight matrix) rather than a plain F32 pointer -- nbytes is the row-major
 * [out_dim,in_dim] quantized footprint, the same formula resident_qtensor
 * itself already validated at manifest-parse time. */
static GpuQTensor gpu_qtensor_off(const RealModel *m, const QTensor *q, const char *what) {
    size_t nbytes = quants_row_bytes(q->ggml_type, q->in_dim) * (size_t)q->out_dim;
    GpuQTensor g;
    g.ggml_type = q->ggml_type;
    g.w_off = gpu_off_of(m, q->data, nbytes, what);
    g.out_dim = q->out_dim;
    g.in_dim = q->in_dim;
    return g;
}

/* Builds one layer's RealLayerGpu from its already-resolved RealLayer
 * (m->layers[layer]) -- every dim/type/stride is copied straight from the
 * fields real_load's own per-layer loop already validated; this function
 * only adds the GPU byte-offset (+bounds/alignment check) for every tensor
 * an actual GPU kernel dispatch will read. */
static void real_gpu_layer_build(RealModel *m, int layer) {
    const Config *cfg = &m->cfg;
    const RealLayer *rl = &m->layers[layer];
    RealLayerGpu *g = &m->gpu_layers[layer];
    int is_sliding = cfg->layer_is_sliding[layer];
    int hidden = cfg->hidden_size;
    int Dh = is_sliding ? cfg->swa_head_dim : cfg->head_dim;
    int num_kv = is_sliding ? cfg->swa_num_key_value_heads : cfg->num_key_value_heads;
    int kv_dim = num_kv * Dh;
    int rel_extent = is_sliding ? cfg->sliding_window_size : cfg->rel_extent;
    int K = cfg->conv_kernel_size;

    g->attn_norm_off = gpu_off_of(m, rl->attn_norm, (size_t)hidden * sizeof(float), "attn_norm.weight");
    g->mlp_norm_off  = gpu_off_of(m, rl->mlp_norm,  (size_t)hidden * sizeof(float), "ffn_norm.weight");
    g->q_norm_off    = gpu_off_of(m, rl->q_norm,    (size_t)Dh * sizeof(float), "attn_q_norm.weight");
    g->k_norm_off    = gpu_off_of(m, rl->k_norm,    (size_t)Dh * sizeof(float), "attn_k_norm.weight");
    g->rel_proj_off  = gpu_off_of(m, rl->rel_proj,
                                   (size_t)cfg->d_rel * (size_t)rel_extent * sizeof(float), "attn_rel_proj.weight");
    g->k_sconv_off = gpu_off_of(m, rl->k_sconv_w, (size_t)kv_dim * (size_t)K * sizeof(float), "shortconv_k.weight");
    g->v_sconv_off = gpu_off_of(m, rl->v_sconv_w, (size_t)kv_dim * (size_t)K * sizeof(float), "shortconv_v.weight");
    g->attn_sconv_off = gpu_off_of(m, rl->attn_sconv_w, (size_t)hidden * (size_t)K * sizeof(float),
                                    "shortconv_attn.weight");
    g->mlp_sconv_off  = gpu_off_of(m, rl->mlp_sconv_w,  (size_t)hidden * (size_t)K * sizeof(float),
                                    "shortconv_mlp.weight");

    g->wq = gpu_qtensor_off(m, &rl->wq, "attn_q.weight");
    g->wk = gpu_qtensor_off(m, &rl->wk, "attn_k.weight");
    g->wv = gpu_qtensor_off(m, &rl->wv, "attn_v.weight");
    g->wr = gpu_qtensor_off(m, &rl->wr, "attn_r.weight");
    g->wo = gpu_qtensor_off(m, &rl->wo, "attn_output.weight");

    if (!cfg->layer_is_sparse[layer]) {
        g->dense_gate = gpu_qtensor_off(m, &rl->dense_gate, "ffn_gate.weight");
        g->dense_up   = gpu_qtensor_off(m, &rl->dense_up,   "ffn_up.weight");
        g->dense_w2   = gpu_qtensor_off(m, &rl->dense_w2,   "ffn_down.weight");
    } else {
        int n_total = cfg->n_routed_experts + cfg->n_shared_experts;
        g->router_w_off = gpu_off_of(m, rl->router_w, (size_t)n_total * (size_t)hidden * sizeof(float),
                                      "ffn_gate_inp.weight");
        g->shared_gate0 = gpu_qtensor_off(m, &rl->shared_gate0, "ffn_gate_shexp.weight");
        g->shared_up0   = gpu_qtensor_off(m, &rl->shared_up0,   "ffn_up_shexp.weight");
        g->shared_w2_0  = gpu_qtensor_off(m, &rl->shared_w2_0,  "ffn_down_shexp.weight");
        g->shared_gate_stride = rl->shared_gate_stride;
        g->shared_up_stride   = rl->shared_up_stride;
        g->shared_w2_stride   = rl->shared_w2_stride;

        /* gpu_qtensor_off above only validated stack index 0's byte range;
         * real_mlp_moe_forward_gpu's dispatch adds s*stride for every other
         * stack index (s in [1,n_shared)) before handing that offset to a
         * GPU kernel -- Task 4's hardening carry requires bounds+alignment
         * on every offset a dispatch actually uses, so validate the OTHER
         * stacks here too, at load time, rather than trusting the stride
         * arithmetic to stay in bounds at dispatch time. */
        size_t gate_nbytes = quants_row_bytes(g->shared_gate0.ggml_type, g->shared_gate0.in_dim) *
                             (size_t)g->shared_gate0.out_dim;
        size_t up_nbytes = quants_row_bytes(g->shared_up0.ggml_type, g->shared_up0.in_dim) *
                           (size_t)g->shared_up0.out_dim;
        size_t w2_nbytes = quants_row_bytes(g->shared_w2_0.ggml_type, g->shared_w2_0.in_dim) *
                           (size_t)g->shared_w2_0.out_dim;
        for (int s = 1; s < cfg->n_shared_experts; s++) {
            size_t goff = g->shared_gate0.w_off + (size_t)s * (size_t)g->shared_gate_stride;
            size_t uoff = g->shared_up0.w_off + (size_t)s * (size_t)g->shared_up_stride;
            size_t woff = g->shared_w2_0.w_off + (size_t)s * (size_t)g->shared_w2_stride;
            if (goff % 2 != 0 || goff + gate_nbytes > m->res_size)
                die("metal: ffn_gate_shexp.weight stack %d [%zu,%zu) exceeds resident.bin size %zu bytes "
                    "or is oddly aligned", s, goff, goff + gate_nbytes, m->res_size);
            if (uoff % 2 != 0 || uoff + up_nbytes > m->res_size)
                die("metal: ffn_up_shexp.weight stack %d [%zu,%zu) exceeds resident.bin size %zu bytes "
                    "or is oddly aligned", s, uoff, uoff + up_nbytes, m->res_size);
            if (woff % 2 != 0 || woff + w2_nbytes > m->res_size)
                die("metal: ffn_down_shexp.weight stack %d [%zu,%zu) exceeds resident.bin size %zu bytes "
                    "or is oddly aligned", s, woff, woff + w2_nbytes, m->res_size);
        }
    }
}

/* --------------------------------- real_load --------------------------------- */

static RealModel real_load(const char *config_path, const char *manifest_path,
                            const char *index_path, const char *tokenizer_path,
                            size_t expert_cache_budget_bytes, int verbose_cache,
                            int expert_io_pagecache) {
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
        /* Task 11 A/B: pagecache mode leaves this fd's default (buffered,
         * page-cache-backed) I/O policy alone -- expert_cache_get fires an
         * F_RDADVISE readahead hint per region instead (ds4's alternative).
         * Default stays F_NOCACHE (direct I/O) per Global Constraints. */
        if (!expert_io_pagecache && fcntl(fd, F_NOCACHE, 1) != 0)
            die("fcntl F_NOCACHE %s: %s", partpath, strerror(errno));
        m.part_fds[i] = fd;
        m.part_sizes[i] = (int64_t)pst.st_size;
    }
    free(gguf_dir);

    /* Bounds-check every streamed expert slot against its part's actual
     * size -- the streamed-tensor analogue of manifest_load's res_size
     * check above -- and cross-check each slot's nbytes against the shape
     * config.json implies (gate/up: hidden->moe_inter, down: moe_inter->
     * hidden), the streamed-tensor analogue of resident_qtensor's shape
     * check. Entries within one (layer, slot) share type+nbytes by
     * construction, so checking expert 0 would suffice, but looping over
     * every (layer, slot, expert) triple is simpler code and still cheap
     * (~49k integer compares, load-time only). */
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
                size_t want = (s == 2)
                    ? quants_row_bytes(sl->ggml_type, cfg->moe_intermediate_size) * (size_t)cfg->hidden_size
                    : quants_row_bytes(sl->ggml_type, cfg->hidden_size) * (size_t)cfg->moe_intermediate_size;
                if ((size_t)sl->nbytes != want)
                    die("%s: layer %d slot '%s' expert %d: nbytes %lld disagrees with shape/type (want %zu)",
                        index_path, layer, slot_names[s], e, (long long)sl->nbytes, want);
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

    /* --metal already ran (main() initializes the GPU runtime before
     * calling real_load): wrap resident.bin's existing mmap zero-copy, then
     * (Task 9) build the per-layer GPU offset table + the logits tensor's
     * GPU descriptor directly from the RealLayer/QTensor fields the loop
     * above just resolved -- gpu_off_of bounds/alignment-checks every one
     * of them again here, dying loudly at load if anything is wrong rather
     * than corrupting a dispatch later. */
    if (sepia_gpu_available()) {
        SepiaGpuBuf *gbuf = sepia_gpu_wrap_mmap(m.res_base, m.res_size);
        if (!gbuf) die("metal: failed to wrap resident.bin (%zu bytes) as a Metal buffer", m.res_size);
        m.gpu_res_buf = gbuf;
        fprintf(stderr, "sepia: metal: resident.bin wrapped (%.2f GB)\n", (double)m.res_size / 1e9);

        m.gpu_layers = xcalloc((size_t)cfg->num_hidden_layers, sizeof(RealLayerGpu));
        for (int i = 0; i < cfg->num_hidden_layers; i++) real_gpu_layer_build(&m, i);
        m.gpu_unembed = gpu_qtensor_off(&m, &m.unembed, "output.weight");
        fprintf(stderr, "sepia: metal: real-model GPU offset table built (%d layers)\n", cfg->num_hidden_layers);

        /* Task 10: the routed-expert GPU cache. m.idx's pointer members
         * (by_layer etc.) are stable heap allocations independent of how
         * many times the enclosing RealModel gets copied, so reading VALUE
         * fields off &m.idx here (never storing that address itself) is
         * safe despite real_load's own return-by-value. */
        expert_store_init(&m.expert_store, &m.idx, expert_cache_budget_bytes, cfg->num_experts_per_tok,
                           verbose_cache, expert_io_pagecache);

        /* Task 11: the loader thread pool (persistent, created once here --
         * see its own header comment further down for the full design). A
         * process-wide singleton: safe because real_load itself only ever
         * runs once per process (main() calls it exactly once on the
         * --real path), matching m.expert_store's own once-per-process
         * lifetime. */
        expert_loader_pool_start();
    }

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

/* ==================== Task 11: expert loader thread pool ================= *
 * A small, persistent pool of pthread workers that do ONLY blocking file I/O
 * (pread into a cache slot's slab memory) and never touch Metal -- src/
 * sepia_metal.m's own thread-safety comment (Task 3) requires this: g_device/
 * g_queue/g_batch_cb/g_batch_enc/g_pso_cache are plain globals with no
 * locking, single-thread-only. This pool lives entirely in sepia.c and never
 * calls a single sepia_gpu_* entry point. The main thread hands off work by
 * pushing a job onto a mutex-protected queue and signaling a "work available"
 * condvar; a worker pulls the job, preads it, then signals a "job done"
 * condvar the main thread blocks on (pthread_cond_wait only -- no polling,
 * no spin, matching the Global Constraints threading rule).
 *
 * Design choice: MTLSharedEvent vs a plain condvar handoff. The plan
 * mentions MTLSharedEvent as one way to let a loader-thread-side "install
 * done" signal interact with GPU encode ordering. This implementation does
 * NOT use it: the loader thread would still have to relay its completion
 * back to the main thread for the main thread to do the actual Metal-side
 * signal/encode (since the loader thread must never touch Metal), which
 * means a plain pthread condvar handoff is required either way -- adding
 * MTLSharedEvent on top would only buy a way for the GPU itself to block on
 * a future value, which nothing here needs: the main thread already has a
 * concrete synchronous point (right before it encodes the dispatch that
 * reads a given expert's slot) where it needs to know "is this data ready
 * yet", and a plain blocking wait against the job's own `done` flag answers
 * that with no extra machinery. Simpler and equally correct; see the task
 * report for the write-up this comment summarizes.
 *
 * Job granularity: one job == one expert's gate+up+down reads (all three
 * must land before that expert's matvec_q dispatches are safe to encode).
 * Queue capacity is bounded to SEPIA_MOE_MAX_TOPK -- at most one sparse
 * layer's worth of misses is ever in flight at a time (real_mlp_moe_forward_
 * gpu always drains every job it submits, via expert_loader_wait, before
 * that layer's SYNC #2 returns -- see its own comments). */
#define SEPIA_LOADER_THREADS 4     /* docs/ssd-bench.md's own finding: F_NOCACHE
                                     * throughput on this machine saturates at
                                     * 4 concurrent requests (11.7/13.3/13.4
                                     * GB/s at threads=1/8/16) -- more workers
                                     * would add contention, not throughput. */
#define SEPIA_LOADER_QUEUE_CAP SEPIA_MOE_MAX_TOPK

typedef struct {
    int fd;
    void *dst;
    size_t nbytes;
    int64_t offset;
    const char *what;
} LoaderRead;

/* Stack-allocated by the caller (real_mlp_moe_forward_gpu keeps an array of
 * these, one per topk slot, alive for the whole duration of its own call --
 * see the comment there) -- the pool only ever stores POINTERS to these, so
 * no heap allocation/ownership question exists here. */
typedef struct ExpertLoadJob {
    LoaderRead reads[3];
    int n_reads;
    int done;      /* 0 = queued/in-progress, 1 = complete; read/written only
                     * under g_loader_pool.mu, so no atomic/volatile needed. */
} ExpertLoadJob;

typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t cv_work;   /* workers wait on this while the queue is empty */
    pthread_cond_t cv_done;   /* waiters (expert_loader_wait) wait on this */
    ExpertLoadJob *queue[SEPIA_LOADER_QUEUE_CAP];
    int qhead, qtail, qcount;
    int shutdown;
    pthread_t workers[SEPIA_LOADER_THREADS];
    int started;
} ExpertLoaderPool;

static ExpertLoaderPool g_loader_pool;

/* ds4's page-cache A/B alternative (Global Constraints; Task 11 A/Bs this
 * against the F_NOCACHE default): a best-effort hint that lets the kernel
 * start reading this exact byte range asynchronously well before the actual
 * pread happens. Fired from the MAIN thread at slot-resolution time (see
 * expert_cache_get below), NOT from the worker that will eventually perform
 * the read -- that's a deliberate choice: it gives the kernel a real head
 * start across however long it takes this token's OTHER selected experts to
 * finish resolving plus however long the job sits in the queue before a
 * worker picks it up, mirroring ds4's own timing (readahead hint issued at
 * cache-miss-detected time, well before the actual read). No-op when
 * F_RDADVISE isn't defined (non-Darwin) or in F_NOCACHE mode (direct I/O
 * bypasses the page cache the hint would warm, so there's nothing to hint).
 * Return value ignored on purpose -- exactly like ds4's own (void) cast: a
 * missed hint only costs a slower first read, never incorrect data. */
static void expert_readahead_hint(int fd, int64_t offset, size_t nbytes) {
#if defined(F_RDADVISE)
    struct radvisory ra;
    ra.ra_offset = (off_t)offset;
    ra.ra_count = (int)nbytes; /* bounded well under INT_MAX -- Task 10's own
                                 * max_slab_bytes arithmetic puts this in the
                                 * low tens of MB at most */
    (void)fcntl(fd, F_RDADVISE, &ra);
#else
    (void)fd; (void)offset; (void)nbytes;
#endif
}

static void expert_load_job_run(ExpertLoadJob *job) {
    for (int i = 0; i < job->n_reads; i++) {
        const LoaderRead *r = &job->reads[i];
        pread_exact(r->fd, r->dst, r->nbytes, r->offset, r->what);
    }
}

static void *expert_loader_worker_main(void *arg) {
    ExpertLoaderPool *pool = (ExpertLoaderPool *)arg;
    for (;;) {
        pthread_mutex_lock(&pool->mu);
        while (pool->qcount == 0 && !pool->shutdown) pthread_cond_wait(&pool->cv_work, &pool->mu);
        if (pool->qcount == 0 && pool->shutdown) {
            pthread_mutex_unlock(&pool->mu);
            return NULL;
        }
        ExpertLoadJob *job = pool->queue[pool->qhead];
        pool->qhead = (pool->qhead + 1) % SEPIA_LOADER_QUEUE_CAP;
        pool->qcount--;
        pthread_mutex_unlock(&pool->mu);

        /* The actual blocking file I/O -- the ONLY thing this thread ever
         * does. No sepia_gpu_* call is reachable from here. */
        expert_load_job_run(job);

        pthread_mutex_lock(&pool->mu);
        job->done = 1;
        pthread_cond_broadcast(&pool->cv_done);
        pthread_mutex_unlock(&pool->mu);
    }
}

/* Created once, at --metal --real startup (real_load, right after
 * expert_store_init). Idempotent (a second call is a no-op) so it stays
 * safe even if a future caller adds another real_load path. */
static void expert_loader_pool_start(void) {
    if (g_loader_pool.started) return;
    memset(&g_loader_pool, 0, sizeof(g_loader_pool));
    if (pthread_mutex_init(&g_loader_pool.mu, NULL) != 0) die("metal: loader-pool: mutex init failed");
    if (pthread_cond_init(&g_loader_pool.cv_work, NULL) != 0) die("metal: loader-pool: cv_work init failed");
    if (pthread_cond_init(&g_loader_pool.cv_done, NULL) != 0) die("metal: loader-pool: cv_done init failed");
    for (int i = 0; i < SEPIA_LOADER_THREADS; i++) {
        if (pthread_create(&g_loader_pool.workers[i], NULL, expert_loader_worker_main, &g_loader_pool) != 0)
            die("metal: loader-pool: pthread_create failed for worker %d", i);
    }
    g_loader_pool.started = 1;
}

/* Signals shutdown, wakes every worker (broadcast -- they're all blocked on
 * cv_work), joins them, then tears the pool down. Called once at process
 * exit, alongside expert_store_shutdown. */
static void expert_loader_pool_shutdown(void) {
    if (!g_loader_pool.started) return;
    pthread_mutex_lock(&g_loader_pool.mu);
    g_loader_pool.shutdown = 1;
    pthread_cond_broadcast(&g_loader_pool.cv_work);
    pthread_mutex_unlock(&g_loader_pool.mu);
    for (int i = 0; i < SEPIA_LOADER_THREADS; i++) pthread_join(g_loader_pool.workers[i], NULL);
    pthread_mutex_destroy(&g_loader_pool.mu);
    pthread_cond_destroy(&g_loader_pool.cv_work);
    pthread_cond_destroy(&g_loader_pool.cv_done);
    memset(&g_loader_pool, 0, sizeof(g_loader_pool));
}

/* Queues `job` (caller-owned storage, see the ExpertLoadJob comment) and
 * returns immediately -- does NOT wait for it to run. */
static void expert_loader_submit(ExpertLoadJob *job) {
    job->done = 0;
    pthread_mutex_lock(&g_loader_pool.mu);
    if (g_loader_pool.qcount >= SEPIA_LOADER_QUEUE_CAP)
        die("metal: loader-pool: queue overflow (>%d in-flight jobs)", SEPIA_LOADER_QUEUE_CAP);
    g_loader_pool.queue[g_loader_pool.qtail] = job;
    g_loader_pool.qtail = (g_loader_pool.qtail + 1) % SEPIA_LOADER_QUEUE_CAP;
    g_loader_pool.qcount++;
    pthread_cond_signal(&g_loader_pool.cv_work);
    pthread_mutex_unlock(&g_loader_pool.mu);
}

/* Blocks (pthread_cond_wait, no polling) until `job` is done. A job that
 * finished before this call returns immediately (the while-loop re-checks
 * job->done itself, so no lost-wakeup window exists between submit and
 * wait). Safe to call on a job that's already done. */
static void expert_loader_wait(ExpertLoadJob *job) {
    pthread_mutex_lock(&g_loader_pool.mu);
    while (!job->done) pthread_cond_wait(&g_loader_pool.cv_done, &g_loader_pool.mu);
    pthread_mutex_unlock(&g_loader_pool.mu);
}

/* Task 10/11: (layer,expert) -> resident GPU slot, installing on a miss.
 * Slot BOOKKEEPING (table/LRU/safe_gen/mlock) is still done synchronously,
 * right here, on the caller's thread -- only the miss install's gate/up/
 * down byte copy is deferred to the loader pool (Task 11). Callers must not
 * have a GPU batch open when calling this (matching real_expert_ffn's own
 * call site convention) -- the resolve phase (this function, called once
 * per selected expert) always finishes before real_mlp_moe_forward_gpu
 * opens its routed-expert dispatch batch. `idx`/`part_fds` are explicit
 * parameters (not stored on `st`) for the same return-by-value safety
 * reason RealExperts documents: the caller always passes its own stable
 * RealModel's &m->idx/m->part_fds.
 *
 * `job_storage` is caller-owned scratch (one ExpertLoadJob per topk slot,
 * kept alive on the caller's stack for the whole call -- see real_mlp_moe_
 * forward_gpu). On a HIT, `*out_job` is set to NULL: the data is already
 * resident, nothing to wait for, and the caller must not touch
 * `job_storage` (it and its returned slot are otherwise independent). On a
 * MISS, `*out_job` is set to `job_storage` itself (already submitted to the
 * loader pool by the time this function returns) -- the caller MUST call
 * expert_loader_wait(*out_job) before reading (or encoding a GPU dispatch
 * that reads) this slot's memory, but may defer that wait arbitrarily to
 * overlap it with other work in the meantime; that deferral is the whole
 * point of Task 11. */
static ExpertCacheSlot *expert_cache_get(ExpertGpuStore *st, const ExpertIndex *idx, const int *part_fds,
                                          int layer, int expert, ExpertLoadJob *job_storage,
                                          ExpertLoadJob **out_job) {
    if (layer < 0 || layer >= st->n_layers_alloc || expert < 0 || expert >= st->n_experts)
        die("metal: expert-cache: (layer=%d,expert=%d) out of range", layer, expert);
    const MoeLayerIndex *mli = &idx->by_layer[layer];
    if (!mli->gate) die("metal: expert-cache: layer %d has no expert index entry", layer);

    int table_idx = layer * st->n_experts + expert;
    int slot_idx = st->table[table_idx];
    if (slot_idx >= 0) {
        st->hits++;
        ExpertCacheSlot *sl = &st->slots[slot_idx];
        /* Task 11 fix: bump safe_gen on a HIT too, not just an install --
         * see the ExpertGpuStore comment above for why this was inert under
         * Task 10's synchronous design but is load-bearing once callers
         * genuinely overlap (this hit's data is about to be read by a
         * dispatch the caller is about to encode; that dispatch completes
         * by AT LEAST the layer's next real_gpu_end(), i.e. gen_completed+1
         * at this exact moment). */
        sl->safe_gen = st->gen_completed + 1;
        expert_cache_lru_unlink(st, slot_idx);
        expert_cache_lru_push_head(st, slot_idx);
        *out_job = NULL;
        return sl;
    }
    st->misses++;

    int new_idx = expert_cache_evict_or_get_free(st);
    ExpertCacheSlot *sl = &st->slots[new_idx];
    if (sl->layer >= 0) st->table[sl->layer * st->n_experts + sl->expert] = -1;

    const ExpertSlot *ge = &mli->gate[expert];
    const ExpertSlot *ue = &mli->up[expert];
    const ExpertSlot *de = &mli->down[expert];

    void *slab_base = sepia_gpu_host_ptr(st->slabs[sl->slab_idx]);
    uint8_t *gate_dst = (uint8_t *)slab_base + sl->base_off;
    uint8_t *up_dst   = (uint8_t *)slab_base + sl->base_off + st->region_bytes;
    uint8_t *down_dst = (uint8_t *)slab_base + sl->base_off + 2 * st->region_bytes;

    if (st->io_pagecache) {
        expert_readahead_hint(part_fds[ge->part_idx], ge->abs_offset, (size_t)ge->nbytes);
        expert_readahead_hint(part_fds[ue->part_idx], ue->abs_offset, (size_t)ue->nbytes);
        expert_readahead_hint(part_fds[de->part_idx], de->abs_offset, (size_t)de->nbytes);
    }

    /* Task 11: hand the actual byte copy to the loader pool instead of
     * pread'ing synchronously here -- the caller decides when it actually
     * needs to wait (right before encoding a dispatch that reads this
     * slot). */
    job_storage->n_reads = 3;
    job_storage->reads[0] = (LoaderRead){part_fds[ge->part_idx], gate_dst, (size_t)ge->nbytes, ge->abs_offset,
                                          "expert gate slab (gpu cache)"};
    job_storage->reads[1] = (LoaderRead){part_fds[ue->part_idx], up_dst,   (size_t)ue->nbytes, ue->abs_offset,
                                          "expert up slab (gpu cache)"};
    job_storage->reads[2] = (LoaderRead){part_fds[de->part_idx], down_dst, (size_t)de->nbytes, de->abs_offset,
                                          "expert down slab (gpu cache)"};
    expert_loader_submit(job_storage);
    *out_job = job_storage;

    sl->ggml_type_gate = ge->ggml_type;
    sl->ggml_type_up   = ue->ggml_type;
    sl->ggml_type_down = de->ggml_type;
    sl->layer = layer;
    sl->expert = expert;
    /* This slot's first GPU read is encoded in the batch the caller opens
     * right after this function returns -- that batch completes under
     * generation gen_completed+1, so it isn't safe to evict until then.
     * Unaffected by making the byte copy itself async: the copy is
     * guaranteed complete (expert_loader_wait) before that dispatch is even
     * encoded, and eviction only ever runs on this same host thread, never
     * concurrently with the copy. */
    sl->safe_gen = st->gen_completed + 1;

    /* mlock just pins whatever physical pages currently back this VA range
     * -- it has no ordering dependency on the (now-deferred) byte copy, so
     * doing it here, before the loader pool has necessarily finished
     * writing, is exactly as correct as doing it after. */
    expert_slot_mlock_once(st, sl, slab_base);

    st->table[table_idx] = new_idx;
    expert_cache_lru_unlink(st, new_idx);
    expert_cache_lru_push_head(st, new_idx);
    return sl;
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

/* ========================= real model, GPU resident path (Task 9) ========= *
 * real_forward_chunk_gpu is the --metal twin of real_forward_chunk above:
 * same math (mirrors decoder_layer_forward/attn_forward_chunk/
 * mlp_dense_forward/mlp_moe_forward exactly), but every resident weight is
 * read straight out of the wrapped gpu_res_buf via the RealLayerGpu offset
 * table (no CPU dequant into the arena -- real_fill_layer/the arena are
 * bypassed entirely in this path) and the KV cache + sconv histories are
 * persistent GPU buffers (GpuCache) instead of host arrays, so attention
 * never re-uploads history the way the tiny GPU path (Task 8) does.
 *
 * Batching: a single logical batch (sepia_gpu_begin/.../end) spans the
 * WHOLE forward-chunk call, paused (end()+begin()) only at the two points
 * the host genuinely needs to read GPU-computed data:
 *   1. every layer's attention block, once, for the tau precision fix
 *      (Metal has no double type, and q_scaled must equal the CPU's
 *      `(float)((double)q * tau)` bit-for-bit -- see real_attn_forward_
 *      chunk_gpu below);
 *   2. every SPARSE layer's MoE block, once per token, to read back router
 *      logits (needed for moe_route_select, host-side) and the mlp-normed
 *      activation (needed as real_expert_ffn's `x`, still CPU-streamed this
 *      task) -- the "expert install boundary" from the plan's sync budget.
 * Dense layers therefore cost exactly 1 sync (tau); sparse layers cost 2
 * (tau + router/expert). Cache writes and sconv-history rolls are GPU-side
 * (sepia_gpu_copy / sepia_gpu_sconv_hist_roll) and need no sync at all --
 * see task-9-report.md for the measured per-token sync count. */

static SepiaGpuBuf *real_gpu_scratch(size_t nfloats) {
    SepiaGpuBuf *b = sepia_gpu_alloc(sizeof(float) * (nfloats ? nfloats : 1), 0);
    if (!b) die("metal: real: scratch GPU buffer alloc failed (%zu floats)", nfloats);
    return b;
}

static SepiaGpuBuf *gpu_zeroed_alloc(size_t nfloats) {
    SepiaGpuBuf *b = sepia_gpu_alloc(sizeof(float) * (nfloats ? nfloats : 1), 0);
    if (!b) die("metal: real: GPU cache buffer alloc failed (%zu floats)", nfloats);
    if (nfloats) memset(sepia_gpu_host_ptr(b), 0, sizeof(float) * nfloats);
    return b;
}

/* Returns a usable [dim]-float, offset-0 operand for row t of a [T,dim]
 * multi-row buffer. Row 0 always sits at byte offset 0, so t==0 returns
 * `multi` directly with ZERO extra dispatches -- this is what keeps the
 * T==1 decode hot path free of any row-copy overhead (T>1 only happens
 * during prefill, a one-time cost). t>0 dispatches a GPU-side copy into the
 * caller-owned, reusable `scratch` buffer -- no host round trip either
 * way. */
static SepiaGpuBuf *real_gpu_row_in(SepiaGpuBuf *multi, int64_t t, int64_t dim, SepiaGpuBuf *scratch) {
    if (t == 0) return multi;
    if (!sepia_gpu_copy(multi, (size_t)(t * dim) * sizeof(float), scratch, 0, dim))
        die("metal: real: row-in copy dispatch failed");
    return scratch;
}

/* Writes a [dim]-float row (e.g. a matvec_q's single-vector output) into row
 * t of a [T,dim] multi-row destination -- only ever called for t>0 (t==0's
 * dispatch already targets `multi` directly, see the call sites below). */
static void real_gpu_row_out(SepiaGpuBuf *srcrow, SepiaGpuBuf *multi, int64_t t, int64_t dim) {
    if (!sepia_gpu_copy(srcrow, 0, multi, (size_t)(t * dim) * sizeof(float), dim))
        die("metal: real: row-out copy dispatch failed");
}

/* Task 10: every real-model-path sepia_gpu_end() goes through this wrapper
 * instead of calling it directly, so the expert cache's in-flight
 * generation counter (m->expert_store.gen_completed) advances at exactly
 * the ds4-anchored point the design calls for ("incremented at
 * sepia_gpu_end()") -- see expert_cache_evict_or_get_free's header comment
 * for what this buys (defensive groundwork for Task 11, inert today under
 * the synchronous per-call design). */
static int real_gpu_end(RealModel *m) {
    int ok = sepia_gpu_end();
    if (ok) m->expert_store.gen_completed++;
    return ok;
}

/* Dispatches sepia_gpu_matvec_q for token row t of a [T,in_dim] input,
 * writing into row t of a [T,out_dim] output -- xrow/yrow are reusable
 * scratch the caller keeps alive across its whole t loop. */
static void real_gpu_matvec_q_row(SepiaGpuBuf *gres, const GpuQTensor *w,
                                   SepiaGpuBuf *x_multi, int64_t t, SepiaGpuBuf *y_multi,
                                   SepiaGpuBuf *xrow, SepiaGpuBuf *yrow, const char *what) {
    SepiaGpuBuf *xt = real_gpu_row_in(x_multi, t, w->in_dim, xrow);
    SepiaGpuBuf *yt = (t == 0) ? y_multi : yrow;
    if (!sepia_gpu_matvec_q(w->ggml_type, gres, w->w_off, xt, yt, w->out_dim, w->in_dim))
        die("metal: real: matvec_q '%s' dispatch failed", what);
    if (t != 0) real_gpu_row_out(yt, y_multi, t, w->out_dim);
}

/* Persistent per-layer GPU KV cache + sconv histories, sized/zeroed exactly
 * like cache_create_cfg's CPU counterpart (same per-layer-type kv_dim
 * formula, same Km1 history depth). */
static GpuCache *gpu_cache_create_cfg(const Config *cfg, int cap) {
    GpuCache *c = xmalloc(sizeof(GpuCache));
    c->cap = cap;
    c->num_layers = cfg->num_hidden_layers;
    c->layers = xcalloc((size_t)c->num_layers, sizeof(GpuLayerCache));
    int Km1 = cfg->conv_kernel_size - 1;
    int hidden = cfg->hidden_size;
    for (int i = 0; i < c->num_layers; i++) {
        GpuLayerCache *lc = &c->layers[i];
        int is_sliding = cfg->layer_is_sliding[i];
        int kv_dim = is_sliding ? cfg->swa_num_key_value_heads * cfg->swa_head_dim
                                : cfg->num_key_value_heads * cfg->head_dim;
        lc->kv_dim = kv_dim;
        lc->k = gpu_zeroed_alloc((size_t)cap * (size_t)kv_dim);
        lc->v = gpu_zeroed_alloc((size_t)cap * (size_t)kv_dim);
        lc->k_hist = gpu_zeroed_alloc((size_t)Km1 * (size_t)kv_dim);
        lc->v_hist = gpu_zeroed_alloc((size_t)Km1 * (size_t)kv_dim);
        lc->attn_hist = gpu_zeroed_alloc((size_t)Km1 * (size_t)hidden);
        lc->mlp_hist = gpu_zeroed_alloc((size_t)Km1 * (size_t)hidden);
        lc->len = 0;
    }
    return c;
}

static void gpu_cache_free(GpuCache *c) {
    if (!c) return;
    for (int i = 0; i < c->num_layers; i++) {
        GpuLayerCache *lc = &c->layers[i];
        sepia_gpu_free(lc->k);
        sepia_gpu_free(lc->v);
        sepia_gpu_free(lc->k_hist);
        sepia_gpu_free(lc->v_hist);
        sepia_gpu_free(lc->attn_hist);
        sepia_gpu_free(lc->mlp_hist);
    }
    free(c->layers);
    free(c);
}

/* GPU-resident twin of attn_forward_chunk. x_normed/out are [T,hidden]. The
 * ONE sync this function performs is the tau precision fix (mandate carry
 * from Task 8's review): q_scaled must equal `(float)((double)q_normed *
 * tau)` bit-for-bit, matching attn_forward_chunk's own formula -- Metal has
 * no double type, so this needs a host round trip. Every other step
 * (projections, sconv+history-roll, per-head norms, cache write,
 * rel-project, banded attention, wo) is GPU-only, no readback. */
static void real_attn_forward_chunk_gpu(RealModel *m, const RealLayerGpu *g, const Config *cfg,
                                         int is_sliding, GpuLayerCache *lc,
                                         SepiaGpuBuf *x_normed, int T, int start_pos, SepiaGpuBuf *out) {
    SepiaGpuBuf *gres = (SepiaGpuBuf *)m->gpu_res_buf;
    int hidden = cfg->hidden_size;
    int H = is_sliding ? cfg->swa_num_attention_heads : cfg->num_attention_heads;
    int Hkv = is_sliding ? cfg->swa_num_key_value_heads : cfg->num_key_value_heads;
    int Dh = is_sliding ? cfg->swa_head_dim : cfg->head_dim;
    int rel_extent = is_sliding ? cfg->sliding_window_size : cfg->rel_extent;
    int q_dim = H * Dh, kv_dim = Hkv * Dh, r_dim = H * cfg->d_rel;
    int K = cfg->conv_kernel_size, d_rel = cfg->d_rel;

    /* 1. projections: wq/wk/wv/wr matvec_q, one dispatch per (weight, t). */
    SepiaGpuBuf *q_raw = real_gpu_scratch((size_t)T * q_dim);
    SepiaGpuBuf *k_raw = real_gpu_scratch((size_t)T * kv_dim);
    SepiaGpuBuf *v_raw = real_gpu_scratch((size_t)T * kv_dim);
    SepiaGpuBuf *r_raw = real_gpu_scratch((size_t)T * r_dim);
    {
        SepiaGpuBuf *xrow = real_gpu_scratch((size_t)hidden);
        SepiaGpuBuf *yrow_q = real_gpu_scratch((size_t)q_dim);
        SepiaGpuBuf *yrow_k = real_gpu_scratch((size_t)kv_dim);
        SepiaGpuBuf *yrow_v = real_gpu_scratch((size_t)kv_dim);
        SepiaGpuBuf *yrow_r = real_gpu_scratch((size_t)r_dim);
        for (int t = 0; t < T; t++) {
            real_gpu_matvec_q_row(gres, &g->wq, x_normed, t, q_raw, xrow, yrow_q, "wq");
            real_gpu_matvec_q_row(gres, &g->wk, x_normed, t, k_raw, xrow, yrow_k, "wk");
            real_gpu_matvec_q_row(gres, &g->wv, x_normed, t, v_raw, xrow, yrow_v, "wv");
            real_gpu_matvec_q_row(gres, &g->wr, x_normed, t, r_raw, xrow, yrow_r, "wr");
        }
        sepia_gpu_free(xrow);
        sepia_gpu_free(yrow_q);
        sepia_gpu_free(yrow_k);
        sepia_gpu_free(yrow_v);
        sepia_gpu_free(yrow_r);
    }

    /* 2. k/v sconv against the layer's persistent GPU history, then roll
     * that history forward on the GPU (no host round trip -- the point of
     * Task 9's sconv_hist_roll kernel over Task 8's host-side design). */
    SepiaGpuBuf *k_sconv = real_gpu_scratch((size_t)T * kv_dim);
    SepiaGpuBuf *v_sconv = real_gpu_scratch((size_t)T * kv_dim);
    if (!sepia_gpu_sconv_off(gres, g->k_sconv_off, lc->k_hist, k_raw, k_sconv, kv_dim, K, T))
        die("metal: real: k sconv dispatch failed");
    if (!sepia_gpu_sconv_off(gres, g->v_sconv_off, lc->v_hist, v_raw, v_sconv, kv_dim, K, T))
        die("metal: real: v sconv dispatch failed");
    if (!sepia_gpu_sconv_hist_roll(lc->k_hist, k_raw, kv_dim, K, T)) die("metal: real: k hist-roll dispatch failed");
    if (!sepia_gpu_sconv_hist_roll(lc->v_hist, v_raw, kv_dim, K, T)) die("metal: real: v hist-roll dispatch failed");
    sepia_gpu_free(k_raw);
    sepia_gpu_free(v_raw);

    /* 3. per-head RMSNorm (no norm on v). */
    SepiaGpuBuf *q_normed = real_gpu_scratch((size_t)T * q_dim);
    SepiaGpuBuf *k_normed = real_gpu_scratch((size_t)T * kv_dim);
    if (!sepia_gpu_rmsnorm_off(gres, g->q_norm_off, q_raw, q_normed, (int64_t)T * H, Dh, (float)cfg->rms_norm_eps))
        die("metal: real: q rmsnorm dispatch failed");
    if (!sepia_gpu_rmsnorm_off(gres, g->k_norm_off, k_sconv, k_normed, (int64_t)T * Hkv, Dh, (float)cfg->rms_norm_eps))
        die("metal: real: k rmsnorm dispatch failed");
    sepia_gpu_free(q_raw);
    sepia_gpu_free(k_sconv);

    /* 4. cache write (normed K, raw-sconv'd V), BEFORE the attention loop
     * reads lc->k/lc->v below -- matches attn_forward_chunk's own ordering
     * (self-attention includes the just-written position). Both are single
     * contiguous-range GPU->GPU copies into the persistent cache at row
     * start_pos -- no host round trip, no reupload of prior history (lc->k/
     * lc->v ARE the full-history buffer banded_attn reads directly, unlike
     * Task 8's tiny path which reuploads the whole history every call). */
    if (!sepia_gpu_copy(k_normed, 0, lc->k, (size_t)start_pos * (size_t)kv_dim * sizeof(float), (int64_t)T * kv_dim))
        die("metal: real: k cache-write dispatch failed");
    if (!sepia_gpu_copy(v_sconv, 0, lc->v, (size_t)start_pos * (size_t)kv_dim * sizeof(float), (int64_t)T * kv_dim))
        die("metal: real: v cache-write dispatch failed");
    sepia_gpu_free(k_normed);
    sepia_gpu_free(v_sconv);
    if (start_pos + T > lc->len) lc->len = start_pos + T;

    /* 5. rel-project (no tau/host dependency -- dispatched before the sync
     * below). */
    SepiaGpuBuf *rel_logits = real_gpu_scratch((size_t)T * (size_t)H * (size_t)rel_extent);
    if (!sepia_gpu_rel_project_off(r_raw, gres, g->rel_proj_off, rel_logits, T, H, d_rel, rel_extent))
        die("metal: real: rel_project dispatch failed");
    sepia_gpu_free(r_raw);

    /* ---- SYNC (attention's one sync per layer): tau precision fix. ---- */
    int have_log_scaling = (!is_sliding) && cfg->has_log_scaling_floor;
    if (!real_gpu_end(m)) die("metal: real: attention tau-sync failed");
    {
        float *qp = (float *)sepia_gpu_host_ptr(q_normed);
        for (int t = 0; t < T; t++) {
            int q_pos = start_pos + t;
            double tau = 1.0;
            if (have_log_scaling) {
                double effective_n = (double)(q_pos + 1);
                double ratio = effective_n / (double)cfg->log_scaling_n_floor;
                if (ratio < 1.0) ratio = 1.0;
                tau = 1.0 + cfg->log_scaling_alpha * log(ratio);
            }
            for (int i = 0; i < q_dim; i++) {
                float *qi = qp + (size_t)t * q_dim + i;
                *qi = (float)((double)(*qi) * tau); /* matches attn_forward_chunk's own formula exactly */
            }
        }
    }
    if (!sepia_gpu_begin()) die("metal: real: failed to resume the batch after the tau sync");

    /* tau/kv_lo/kv_hi host arrays for banded_attn -- same formula as CPU. */
    int64_t *kv_lo = xmalloc(sizeof(int64_t) * (size_t)T);
    int64_t *kv_hi = xmalloc(sizeof(int64_t) * (size_t)T);
    float *tau_arr = xmalloc(sizeof(float) * (size_t)T);
    for (int t = 0; t < T; t++) {
        int q_pos = start_pos + t;
        double tv = 1.0;
        if (have_log_scaling) {
            double effective_n = (double)(q_pos + 1);
            double ratio = effective_n / (double)cfg->log_scaling_n_floor;
            if (ratio < 1.0) ratio = 1.0;
            tv = 1.0 + cfg->log_scaling_alpha * log(ratio);
        }
        tau_arr[t] = (float)tv;
        int64_t kvlo = is_sliding ? (int64_t)q_pos - (int64_t)cfg->sliding_window_size + 1 : 0;
        if (kvlo < 0) kvlo = 0;
        kv_lo[t] = kvlo;
        kv_hi[t] = q_pos;
    }

    /* 6. banded attention over the full persistent cache (lc->k/lc->v
     * already hold [0,start_pos+T) after step 4's write; the buffer's own
     * allocated capacity is >= that, and the kernel never reads past
     * kv_hi). */
    SepiaGpuBuf *attn_concat = real_gpu_scratch((size_t)T * q_dim);
    if (!sepia_gpu_banded_attn(q_normed, lc->k, lc->v, rel_logits, attn_concat, kv_lo, kv_hi, tau_arr,
                                T, H, Hkv, Dh, rel_extent, start_pos, kv_dim, 1.0f / (float)Dh))
        die("metal: real: banded_attn dispatch failed");
    free(kv_lo);
    free(kv_hi);
    free(tau_arr);
    sepia_gpu_free(q_normed);
    sepia_gpu_free(rel_logits);

    /* 7. wo matvec_q, one dispatch per t. */
    {
        SepiaGpuBuf *xrow = real_gpu_scratch((size_t)q_dim);
        SepiaGpuBuf *yrow = real_gpu_scratch((size_t)hidden);
        for (int t = 0; t < T; t++)
            real_gpu_matvec_q_row(gres, &g->wo, attn_concat, t, out, xrow, yrow, "wo");
        sepia_gpu_free(xrow);
        sepia_gpu_free(yrow);
    }
    sepia_gpu_free(attn_concat);
}

/* GPU twin of mlp_dense_forward: gate/up are SEPARATE on-disk quantized
 * tensors in the real GGUF layout (unlike the SafeTensors oracle's
 * interleaved dense_w13), so there is no gather trick to reproduce here --
 * a plain matvec_q per weight, then silu_mul, then matvec_q down, then the
 * global_scale (known before any readback -- always fully GPU-side, see
 * sepia_gpu_scale's header doc). */
static void real_mlp_dense_forward_gpu(RealModel *m, const RealLayerGpu *g, int hidden, int dense_inter,
                                        const RealLayer *rl, SepiaGpuBuf *x, int T, SepiaGpuBuf *out) {
    SepiaGpuBuf *gres = (SepiaGpuBuf *)m->gpu_res_buf;
    SepiaGpuBuf *g_full = real_gpu_scratch((size_t)T * dense_inter);
    SepiaGpuBuf *u_full = real_gpu_scratch((size_t)T * dense_inter);
    {
        SepiaGpuBuf *xrow = real_gpu_scratch((size_t)hidden);
        SepiaGpuBuf *yrow_g = real_gpu_scratch((size_t)dense_inter);
        SepiaGpuBuf *yrow_u = real_gpu_scratch((size_t)dense_inter);
        for (int t = 0; t < T; t++) {
            real_gpu_matvec_q_row(gres, &g->dense_gate, x, t, g_full, xrow, yrow_g, "dense_gate");
            real_gpu_matvec_q_row(gres, &g->dense_up, x, t, u_full, xrow, yrow_u, "dense_up");
        }
        sepia_gpu_free(xrow);
        sepia_gpu_free(yrow_g);
        sepia_gpu_free(yrow_u);
    }

    SepiaGpuBuf *h_full = real_gpu_scratch((size_t)T * dense_inter);
    if (!sepia_gpu_silu_mul(g_full, u_full, h_full, (int64_t)T * dense_inter))
        die("metal: real: dense silu_mul dispatch failed");
    sepia_gpu_free(g_full);
    sepia_gpu_free(u_full);

    SepiaGpuBuf *y_full = real_gpu_scratch((size_t)T * hidden);
    {
        SepiaGpuBuf *hrow = real_gpu_scratch((size_t)dense_inter);
        SepiaGpuBuf *yrow = real_gpu_scratch((size_t)hidden);
        for (int t = 0; t < T; t++)
            real_gpu_matvec_q_row(gres, &g->dense_w2, h_full, t, y_full, hrow, yrow, "dense_w2");
        sepia_gpu_free(hrow);
        sepia_gpu_free(yrow);
    }
    sepia_gpu_free(h_full);

    float gscale = rl->dense_global_scale[0];
    if (!sepia_gpu_scale(gscale, y_full, out, (int64_t)T * hidden)) die("metal: real: dense gscale dispatch failed");
    sepia_gpu_free(y_full);
}

/* GPU twin of mlp_moe_forward for ONE token row t, writing the routed+shared
 * mixed output into out_row[hidden] (a host array -- Task 10 moves this
 * fully onto GPU). Router matvec + shared-expert compute (gate/up/down, no
 * gamma -- see below) run on GPU before any readback, since neither depends
 * on router SELECTION; only after the router-logits sync do we know which
 * routed experts to fetch -- Task 10 streams them through the GPU-resident
 * LRU expert cache (m->expert_store / expert_cache_get) instead of
 * real_expert_ffn's CPU pread+qlinear -- and what each shared expert's
 * mixing weight (gamma) is. Since down_proj is linear, gamma*down_proj(h)
 * == down_proj(gamma*h) -- mlp_moe_forward's own comment notes this
 * equivalence -- so folding gamma in AFTER the (already GPU-computed)
 * shared down_proj output is mathematically identical to the CPU oracle's
 * h*=gamma-before-down_proj order, and lets the expensive matvec_q work
 * happen before the sync instead of after. This function now spans TWO
 * syncs, not one: SYNC #1 (below) publishes router logits + shared outputs
 * so expert slots can be resolved on the CPU before any new batch opens;
 * SYNC #2 (further down) publishes the routed-expert down-projections for
 * the final weighted mix. See the inline "SYNC #1"/"SYNC #2" comments at
 * their call sites for the exact boundaries. */
/* One routed expert's gate/up/silu/down dispatch chain, factored out so both
 * the hit-pass and the miss-pass below (Task 11) share the exact same
 * dispatch code. xt/gbuf/ubuf/hbuf are the caller's scratch buffers, reused
 * across every expert exactly like the shared-expert loop above SYNC #1
 * already does -- safe across a sepia_gpu_flush() command-buffer boundary
 * (used between the two passes) because Metal's default hazard tracking
 * orders access to a given MTLBuffer across command buffers committed to
 * one queue, not just within a single encoder; command buffers submitted to
 * the same MTLCommandQueue are also guaranteed to execute in commit order.
 * That combination is exactly the contract sepia_gpu_flush()'s own header
 * doc describes ("commits... WITHOUT waiting... opens a fresh batch so
 * dispatching can continue") -- this is simply the first real caller to
 * exercise it. */
static void real_mlp_moe_dispatch_routed(RealModel *m, const ExpertCacheSlot *sl, SepiaGpuBuf *xt,
                                          SepiaGpuBuf *gbuf, SepiaGpuBuf *ubuf, SepiaGpuBuf *hbuf,
                                          int moe_inter, int hidden, SepiaGpuBuf **out_row, int j) {
    SepiaGpuBuf *slab = m->expert_store.slabs[sl->slab_idx];
    size_t off_gate = sl->base_off;
    size_t off_up   = sl->base_off + m->expert_store.region_bytes;
    size_t off_down = sl->base_off + 2 * m->expert_store.region_bytes;

    if (!sepia_gpu_matvec_q(sl->ggml_type_gate, slab, off_gate, xt, gbuf, moe_inter, hidden))
        die("metal: real: routed[%d] gate matvec_q dispatch failed", j);
    if (!sepia_gpu_matvec_q(sl->ggml_type_up, slab, off_up, xt, ubuf, moe_inter, hidden))
        die("metal: real: routed[%d] up matvec_q dispatch failed", j);
    if (!sepia_gpu_silu_mul(gbuf, ubuf, hbuf, moe_inter))
        die("metal: real: routed[%d] silu_mul dispatch failed", j);
    *out_row = real_gpu_scratch((size_t)hidden);
    if (!sepia_gpu_matvec_q(sl->ggml_type_down, slab, off_down, hbuf, *out_row, hidden, moe_inter))
        die("metal: real: routed[%d] down matvec_q dispatch failed", j);
}

static void real_mlp_moe_forward_gpu(RealModel *m, const RealLayerGpu *g, const Config *cfg, int layer,
                                      SepiaGpuBuf *x_multi, int64_t t, int hidden, float *out_row) {
    SepiaGpuBuf *gres = (SepiaGpuBuf *)m->gpu_res_buf;
    const RealLayer *rl = &m->layers[layer];
    int n_shared = cfg->n_shared_experts, topk = cfg->num_experts_per_tok;
    int n_total = cfg->n_routed_experts + n_shared;
    int moe_inter = cfg->moe_intermediate_size;

    SepiaGpuBuf *xrow = real_gpu_scratch((size_t)hidden);
    SepiaGpuBuf *xt = real_gpu_row_in(x_multi, t, hidden, xrow);

    SepiaGpuBuf *router_logits_buf = real_gpu_scratch((size_t)n_total);
    if (!sepia_gpu_matvec_off(gres, g->router_w_off, xt, router_logits_buf, n_total, hidden))
        die("metal: real: router matvec dispatch failed");

    SepiaGpuBuf **shared_raw = xmalloc(sizeof(SepiaGpuBuf *) * (size_t)n_shared);
    SepiaGpuBuf *gbuf = real_gpu_scratch((size_t)moe_inter);
    SepiaGpuBuf *ubuf = real_gpu_scratch((size_t)moe_inter);
    SepiaGpuBuf *hbuf = real_gpu_scratch((size_t)moe_inter);
    for (int s = 0; s < n_shared; s++) {
        GpuQTensor gate_s = g->shared_gate0, up_s = g->shared_up0, w2_s = g->shared_w2_0;
        gate_s.w_off += (size_t)s * (size_t)g->shared_gate_stride;
        up_s.w_off += (size_t)s * (size_t)g->shared_up_stride;
        w2_s.w_off += (size_t)s * (size_t)g->shared_w2_stride;

        if (!sepia_gpu_matvec_q(gate_s.ggml_type, gres, gate_s.w_off, xt, gbuf, gate_s.out_dim, gate_s.in_dim))
            die("metal: real: shared[%d] gate matvec_q dispatch failed", s);
        if (!sepia_gpu_matvec_q(up_s.ggml_type, gres, up_s.w_off, xt, ubuf, up_s.out_dim, up_s.in_dim))
            die("metal: real: shared[%d] up matvec_q dispatch failed", s);
        if (!sepia_gpu_silu_mul(gbuf, ubuf, hbuf, moe_inter)) die("metal: real: shared[%d] silu_mul dispatch failed", s);
        shared_raw[s] = real_gpu_scratch((size_t)hidden);
        if (!sepia_gpu_matvec_q(w2_s.ggml_type, gres, w2_s.w_off, hbuf, shared_raw[s], w2_s.out_dim, w2_s.in_dim))
            die("metal: real: shared[%d] down matvec_q dispatch failed", s);
    }

    /* ---- SYNC #1: router logits + shared raw outputs readable off this
     * batch's completion. ---- */
    if (!real_gpu_end(m)) die("metal: real: MoE router/shared readback sync failed");

    const float *router_logits = (const float *)sepia_gpu_host_ptr(router_logits_buf);
    LayerWeights fake_lw = {0}; /* moe_route_select only reads router_bias/router_global_scale */
    fake_lw.router_bias = rl->router_bias;
    fake_lw.router_global_scale = rl->router_global_scale;
    int *topk_idx = xmalloc(sizeof(int) * (size_t)topk);
    int n_sel = topk + n_shared;
    float *weights = xmalloc(sizeof(float) * (size_t)n_sel);
    moe_route_select(cfg, &fake_lw, router_logits, topk_idx, weights);

    /* Task 10/11: routed experts stream through the GPU-resident LRU cache
     * (m->expert_store) instead of real_expert_ffn's per-call CPU
     * pread+qlinear -- real_expert_ffn itself is unchanged and still serves
     * CPU real mode. Resolve every selected expert's slot BEFORE opening a
     * new batch: table/LRU/safe_gen/mlock bookkeeping is pure CPU work that
     * must not straddle an open encoder. Task 11: a miss's actual byte copy
     * is no longer done here -- expert_cache_get hands it to the loader
     * thread pool and returns immediately with pending[j] set to the
     * in-flight job (NULL for a hit, meaning the data is already resident).
     * jobs[]/pending[] live on THIS call's stack for its whole duration --
     * every job this call submits is drained (expert_loader_wait) before
     * this function returns, so their lifetime is never in question. */
    if (topk > SEPIA_MOE_MAX_TOPK)
        die("metal: real: num_experts_per_tok=%d exceeds the GPU expert-cache dispatch's compiled bound (%d)",
            topk, SEPIA_MOE_MAX_TOPK);
    ExpertCacheSlot *cslot[SEPIA_MOE_MAX_TOPK];
    ExpertLoadJob jobs[SEPIA_MOE_MAX_TOPK];
    ExpertLoadJob *pending[SEPIA_MOE_MAX_TOPK];
    for (int j = 0; j < topk; j++)
        cslot[j] = expert_cache_get(&m->expert_store, &m->idx, m->part_fds, layer, topk_idx[j],
                                     &jobs[j], &pending[j]);

    if (!sepia_gpu_begin()) die("metal: real: failed to open the routed-expert dispatch batch");
    SepiaGpuBuf *routed_raw[SEPIA_MOE_MAX_TOPK];

    /* Pass 1: encode every HIT's dispatch now -- its data is already
     * resident, nothing to wait for. This is the GPU work the plan wants
     * running WHILE the loader threads stream in the misses below. */
    for (int j = 0; j < topk; j++) {
        if (pending[j] != NULL) continue; /* miss -- deferred to pass 2 */
        real_mlp_moe_dispatch_routed(m, cslot[j], xt, gbuf, ubuf, hbuf, moe_inter, hidden, &routed_raw[j], j);
    }
    /* Commit pass 1 WITHOUT waiting (fire-and-forget) so the GPU actually
     * starts executing the hit-batch concurrently with the still-in-flight
     * loader-thread preads below -- this is the actual overlap: without this
     * flush, nothing would be committed (and the GPU would sit idle) until
     * AFTER the wait loop below finishes, defeating the whole point. Safe
     * (and cheap) even when pass 1 encoded nothing, e.g. an all-miss step. */
    if (!sepia_gpu_flush()) die("metal: real: routed-expert hit-batch flush failed");

    /* Pass 2: block on each miss's own pread completion, one at a time,
     * only right before encoding ITS dispatch -- a hit or an
     * already-finished prefetch never waits (per-job wait, not "wait for
     * everything"). expert_loader_wait blocks on pthread_cond_wait only --
     * no polling, matching the Global Constraints threading rule. */
    for (int j = 0; j < topk; j++) {
        if (pending[j] == NULL) continue;
        expert_loader_wait(pending[j]);
        real_mlp_moe_dispatch_routed(m, cslot[j], xt, gbuf, ubuf, hbuf, moe_inter, hidden, &routed_raw[j], j);
    }

    /* ---- SYNC #2: routed-expert down-projections readable off this
     * batch's completion. This single end() call also covers pass 1's
     * already-flushed command buffer: MTLCommandQueue executes command
     * buffers submitted to it strictly in commit order, so waiting on the
     * LAST one's completion transitively guarantees every earlier one
     * (including pass 1's) has also completed -- gen_completed only needs
     * to advance once here for every slot this call touched (hit or miss)
     * to be correctly "safe through this generation". ---- */
    if (!real_gpu_end(m)) die("metal: real: routed-expert readback sync failed");

    for (int d = 0; d < hidden; d++) out_row[d] = 0.0f;
    for (int j = 0; j < topk; j++) {
        const float *eraw = (const float *)sepia_gpu_host_ptr(routed_raw[j]);
        float wj = weights[j];
        for (int d = 0; d < hidden; d++) out_row[d] += eraw[d] * wj;
    }
    for (int s = 0; s < n_shared; s++) {
        const float *sraw = (const float *)sepia_gpu_host_ptr(shared_raw[s]);
        float gamma = weights[topk + s];
        for (int d = 0; d < hidden; d++) out_row[d] += sraw[d] * gamma;
    }
    free(topk_idx);
    free(weights);

    for (int j = 0; j < topk; j++) sepia_gpu_free(routed_raw[j]);
    for (int s = 0; s < n_shared; s++) sepia_gpu_free(shared_raw[s]);
    free(shared_raw);
    sepia_gpu_free(router_logits_buf);
    sepia_gpu_free(gbuf);
    sepia_gpu_free(ubuf);
    sepia_gpu_free(hbuf);
    sepia_gpu_free(xrow);

    /* ---- resume the batch for the caller to continue encoding. ---- */
    if (!sepia_gpu_begin()) die("metal: real: failed to resume the batch after the MoE sync");
}

/* GPU twin of decoder_layer_forward: same pre-norm residual wiring, same
 * per-layer sconv history (now GPU-persistent) across calls. */
static void real_decoder_layer_forward_gpu(RealModel *m, int layer, const Config *cfg, GpuLayerCache *lc,
                                            SepiaGpuBuf *x, int T, int start_pos) {
    const RealLayerGpu *g = &m->gpu_layers[layer];
    SepiaGpuBuf *gres = (SepiaGpuBuf *)m->gpu_res_buf;
    int hidden = cfg->hidden_size;
    int K = cfg->conv_kernel_size;
    int is_sliding = cfg->layer_is_sliding[layer];

    SepiaGpuBuf *normed = real_gpu_scratch((size_t)T * hidden);
    if (!sepia_gpu_rmsnorm_off(gres, g->attn_norm_off, x, normed, T, hidden, (float)cfg->rms_norm_eps))
        die("metal: real: attn_norm dispatch failed");

    SepiaGpuBuf *attn_out = real_gpu_scratch((size_t)T * hidden);
    real_attn_forward_chunk_gpu(m, g, cfg, is_sliding, lc, normed, T, start_pos, attn_out);
    sepia_gpu_free(normed);

    SepiaGpuBuf *attn_sconv_out = real_gpu_scratch((size_t)T * hidden);
    if (!sepia_gpu_sconv_off(gres, g->attn_sconv_off, lc->attn_hist, attn_out, attn_sconv_out, hidden, K, T))
        die("metal: real: attn sconv dispatch failed");
    if (!sepia_gpu_sconv_hist_roll(lc->attn_hist, attn_out, hidden, K, T))
        die("metal: real: attn hist-roll dispatch failed");
    sepia_gpu_free(attn_out);
    if (!sepia_gpu_add(x, attn_sconv_out, x, (int64_t)T * hidden)) die("metal: real: attn residual add dispatch failed");
    sepia_gpu_free(attn_sconv_out);

    SepiaGpuBuf *normed2 = real_gpu_scratch((size_t)T * hidden);
    if (!sepia_gpu_rmsnorm_off(gres, g->mlp_norm_off, x, normed2, T, hidden, (float)cfg->rms_norm_eps))
        die("metal: real: mlp_norm dispatch failed");

    SepiaGpuBuf *mlp_out = real_gpu_scratch((size_t)T * hidden);
    if (!cfg->layer_is_sparse[layer]) {
        real_mlp_dense_forward_gpu(m, g, hidden, cfg->dense_intermediate_size, &m->layers[layer], normed2, T, mlp_out);
    } else {
        float *out_row = xmalloc(sizeof(float) * (size_t)hidden);
        float *mlp_out_host = xmalloc(sizeof(float) * (size_t)T * hidden);
        for (int t = 0; t < T; t++) {
            real_mlp_moe_forward_gpu(m, g, cfg, layer, normed2, t, hidden, out_row);
            memcpy(mlp_out_host + (size_t)t * hidden, out_row, sizeof(float) * (size_t)hidden);
        }
        free(out_row);
        /* Host-accumulated MoE output -> mlp_out's Shared storage: a plain
         * memcpy, no dispatch touched mlp_out yet this call, so no extra
         * sync is needed (the batch real_mlp_moe_forward_gpu leaves open on
         * return is exactly the one this memcpy writes into, safely, since
         * nothing has been encoded against mlp_out in it). */
        memcpy(sepia_gpu_host_ptr(mlp_out), mlp_out_host, sizeof(float) * (size_t)T * (size_t)hidden);
        free(mlp_out_host);
    }
    sepia_gpu_free(normed2);

    SepiaGpuBuf *mlp_sconv_out = real_gpu_scratch((size_t)T * hidden);
    if (!sepia_gpu_sconv_off(gres, g->mlp_sconv_off, lc->mlp_hist, mlp_out, mlp_sconv_out, hidden, K, T))
        die("metal: real: mlp sconv dispatch failed");
    if (!sepia_gpu_sconv_hist_roll(lc->mlp_hist, mlp_out, hidden, K, T))
        die("metal: real: mlp hist-roll dispatch failed");
    sepia_gpu_free(mlp_out);
    if (!sepia_gpu_add(x, mlp_sconv_out, x, (int64_t)T * hidden)) die("metal: real: mlp residual add dispatch failed");
    sepia_gpu_free(mlp_sconv_out);
}

/* GPU twin of real_forward_chunk. Embed lookup + embed_norm + final_norm
 * stay on the CPU (single-row-per-token rmsnorm over a [hidden] vector --
 * negligible cost either way, and this keeps the residual stream's one
 * genuinely necessary host round trip -- reading the last layer's output
 * back for final_norm -- as the only non-per-layer sync in this function,
 * plus the logits matvec_q's own readback). */
static void real_forward_chunk_gpu(RealModel *m, GpuCache *gcache, const int *token_ids, int T, int start_pos,
                                    float *logits_out) {
    const Config *cfg = &m->cfg;
    int hidden = cfg->hidden_size;

    float *x_host = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    float *erow = xmalloc(sizeof(float) * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        qrow(&m->embed, token_ids[t], erow);
        rmsnorm(erow, m->embed_norm, hidden, (float)cfg->rms_norm_eps, x_host + (size_t)t * hidden);
    }
    free(erow);

    SepiaGpuBuf *x = real_gpu_scratch((size_t)T * hidden);
    memcpy(sepia_gpu_host_ptr(x), x_host, sizeof(float) * (size_t)T * (size_t)hidden);
    free(x_host);

    if (!sepia_gpu_begin()) die("metal: real: failed to open the forward-chunk batch");
    for (int l = 0; l < cfg->num_hidden_layers; l++)
        real_decoder_layer_forward_gpu(m, l, cfg, &gcache->layers[l], x, T, start_pos);
    if (!real_gpu_end(m)) die("metal: real: final residual-stream readback sync failed");

    const float *x_final = (const float *)sepia_gpu_host_ptr(x);
    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++)
        rmsnorm(x_final + (size_t)t * hidden, m->final_norm, hidden, (float)cfg->rms_norm_eps,
                normed + (size_t)t * hidden);
    sepia_gpu_free(x);

    if (logits_out) {
        int unpadded = cfg->unpadded_vocab_size;
        float mup = (float)cfg->logits_mup_width_multiplier;
        float *h = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
        for (int i = 0; i < T * hidden; i++) h[i] = normed[i] / mup;

        SepiaGpuBuf *hbuf = real_gpu_scratch((size_t)T * hidden);
        memcpy(sepia_gpu_host_ptr(hbuf), h, sizeof(float) * (size_t)T * (size_t)hidden);
        free(h);

        SepiaGpuBuf *gres = (SepiaGpuBuf *)m->gpu_res_buf;
        /* Q4_K matvec_q over only the unpadded rows (architecture-notes.md
         * sec.8) -- same "cap out_dim to the unpadded count" trick
         * real_forward_chunk's CPU path applies to unembed_unpadded. */
        GpuQTensor unembed_unpadded = m->gpu_unembed;
        unembed_unpadded.out_dim = unpadded;
        SepiaGpuBuf *logits_buf = real_gpu_scratch((size_t)T * unpadded);
        SepiaGpuBuf *hrow = real_gpu_scratch((size_t)hidden);
        SepiaGpuBuf *yrow = real_gpu_scratch((size_t)unpadded);
        if (!sepia_gpu_begin()) die("metal: real: failed to open the logits batch");
        for (int t = 0; t < T; t++)
            real_gpu_matvec_q_row(gres, &unembed_unpadded, hbuf, t, logits_buf, hrow, yrow, "logits");
        if (!real_gpu_end(m)) die("metal: real: logits readback sync failed");
        memcpy(logits_out, sepia_gpu_host_ptr(logits_buf), sizeof(float) * (size_t)T * (size_t)unpadded);

        sepia_gpu_free(hbuf);
        sepia_gpu_free(logits_buf);
        sepia_gpu_free(hrow);
        sepia_gpu_free(yrow);
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

/* --verbose-cache: prints this step's hit/miss delta plus the running
 * total, then folds the delta into (*prev_hits, *prev_misses) so the NEXT
 * call reports a fresh delta. A no-op when the store wasn't built verbose
 * (--metal --real without --verbose-cache, or CPU real mode). */
static void expert_cache_report_step(const ExpertGpuStore *st, const char *label,
                                      uint64_t *prev_hits, uint64_t *prev_misses) {
    if (!st->verbose) return;
    uint64_t dh = st->hits - *prev_hits, dm = st->misses - *prev_misses;
    uint64_t dt = dh + dm;
    uint64_t th = st->hits, tm = st->misses, tt = th + tm;
    fprintf(stderr,
            "sepia: metal: expert-cache: %s hits=%llu misses=%llu (%.1f%% hit) "
            "-- total hits=%llu misses=%llu (%.1f%% hit)\n",
            label, (unsigned long long)dh, (unsigned long long)dm, dt ? 100.0 * (double)dh / (double)dt : 0.0,
            (unsigned long long)th, (unsigned long long)tm, tt ? 100.0 * (double)th / (double)tt : 0.0);
    *prev_hits = st->hits;
    *prev_misses = st->misses;
}

/* --metal --real twins of real_generate/real_print_top_logits, identical in
 * every way except the cache type and the forward-chunk call -- see
 * real_forward_chunk_gpu's header doc for what's actually different under
 * the hood (resident weights read via GPU offsets instead of CPU-dequanted
 * into an arena; KV cache + sconv history persistent on GPU). */
static void real_generate_gpu(RealModel *m, const char *prompt, int n_gen) {
    const Config *cfg = &m->cfg;
    int unpadded = cfg->unpadded_vocab_size;

    int32_t *ids = xmalloc(sizeof(int32_t) * (size_t)REAL_MAX_PROMPT_IDS);
    int n_prompt = tokenizer_encode(m->tok, prompt, ids, REAL_MAX_PROMPT_IDS);
    int *ids32 = xmalloc(sizeof(int) * (size_t)n_prompt);
    for (int i = 0; i < n_prompt; i++) ids32[i] = (int)ids[i];

    GpuCache *cache = gpu_cache_create_cfg(cfg, n_prompt + n_gen + 8);

    /* Snapshot the store's CUMULATIVE counts before this call -- on a warm
     * repeat (same process, m->expert_store already populated from a prior
     * real_generate_gpu call), this makes the first reported delta reflect
     * only what THIS call does, not everything accumulated before it. */
    uint64_t prev_hits = m->expert_store.hits, prev_misses = m->expert_store.misses;

    float *logits = xmalloc(sizeof(float) * (size_t)n_prompt * (size_t)unpadded);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    real_forward_chunk_gpu(m, cache, ids32, n_prompt, 0, logits);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = elapsed_ms(t0, t1);
    free(ids32);

    expert_cache_report_step(&m->expert_store, "prefill", &prev_hits, &prev_misses);

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
        real_forward_chunk_gpu(m, cache, &next_id, 1, pos, cur);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        ms = elapsed_ms(t0, t1);
        char label[32];
        snprintf(label, sizeof label, "token %d", i + 1);
        expert_cache_report_step(&m->expert_store, label, &prev_hits, &prev_misses);
        pos++;
    }
    gpu_cache_free(cache);
    free(ids);
    free(cur);
}

static void real_print_top_logits_gpu(RealModel *m, const char *prompt) {
    const Config *cfg = &m->cfg;
    int unpadded = cfg->unpadded_vocab_size;

    int32_t *ids = xmalloc(sizeof(int32_t) * (size_t)REAL_MAX_PROMPT_IDS);
    int n_prompt = tokenizer_encode(m->tok, prompt, ids, REAL_MAX_PROMPT_IDS);
    int *ids32 = xmalloc(sizeof(int) * (size_t)n_prompt);
    for (int i = 0; i < n_prompt; i++) ids32[i] = (int)ids[i];

    GpuCache *cache = gpu_cache_create_cfg(cfg, n_prompt);
    float *logits = xmalloc(sizeof(float) * (size_t)n_prompt * (size_t)unpadded);
    real_forward_chunk_gpu(m, cache, ids32, n_prompt, 0, logits);
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

    uint64_t prev_hits = 0, prev_misses = 0;
    expert_cache_report_step(&m->expert_store, "logits-only prompt", &prev_hits, &prev_misses);

    free(logits);
    free(ids32);
    free(ids);
    gpu_cache_free(cache);
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

/* ------------------------------ gpu selftest -------------------------------- */

/* --gpu-selftest (local-only, needs --metal to have already initialized a
 * live device): exercises sepia_gpu_wrap_mmap/alloc/free/host_ptr/gpu_addr
 * end-to-end, standing in for the fixture-style gates the CPU quant/tokenizer
 * suites use since there is no committed "expected GPU buffer" fixture format.
 *
 * (1) wrap a 3-page mmap'd temp file filled with sequential float values
 *     0..N (not an arbitrary byte pattern -- x += 0.0f is only an identity
 *     for finite floats, and a reinterpreted-arbitrary-bits float can be a
 *     signaling NaN on some hardware), dispatch sepia_touch over only the
 *     first page's floats, and verify via host_ptr that every float in the
 *     whole buffer still reads back exactly as written: the touched page
 *     proves the dispatch really executed against the zero-copy memory
 *     (host_ptr must equal the original base -- no copy happened), the
 *     untouched pages prove the wrap+dispatch didn't corrupt neighboring
 *     memory.
 * (2) wrap_mmap(base+1, ...) (page-alignment violation) must return NULL,
 *     not crash.
 * (3) alloc Shared and Private buffers, check host_ptr NULL-ness, free both.
 *
 * Prints "gpu selftest ok" on success; dies loudly (via die()) otherwise. */
static void run_gpu_selftest(void) {
    if (!sepia_gpu_available()) die("gpu-selftest: requires --metal to have initialized the GPU runtime");

    const size_t page = (size_t)getpagesize();
    const size_t n_pages = 3;
    const size_t total_len = n_pages * page;
    const size_t floats_per_page = page / sizeof(float);
    const size_t n_floats_total = total_len / sizeof(float);

    const char *tmpdir = getenv("TMPDIR");
    if (!tmpdir || !tmpdir[0]) tmpdir = "/tmp";
    char path[1024];
    int pn = snprintf(path, sizeof path, "%s/sepia-gpu-selftest-XXXXXX", tmpdir);
    if (pn < 0 || (size_t)pn >= sizeof path) die("gpu-selftest: tmp path too long");
    int fd = mkstemp(path);
    if (fd < 0) die("gpu-selftest: mkstemp %s: %s", path, strerror(errno));
    if (ftruncate(fd, (off_t)total_len) != 0) die("gpu-selftest: ftruncate: %s", strerror(errno));

    void *base = mmap(NULL, total_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) die("gpu-selftest: mmap: %s", strerror(errno));
    close(fd);
    unlink(path);

    float *fbase = (float *)base;
    for (size_t i = 0; i < n_floats_total; i++) fbase[i] = (float)i;

    SepiaGpuBuf *gbuf = sepia_gpu_wrap_mmap(base, total_len);
    if (!gbuf) die("gpu-selftest: wrap_mmap failed on a valid page-aligned mapping");

    if (!sepia_gpu_selftest_touch(gbuf, floats_per_page))
        die("gpu-selftest: sepia_touch dispatch failed");

    float *readback = (float *)sepia_gpu_host_ptr(gbuf);
    if (!readback) die("gpu-selftest: host_ptr returned NULL for a Shared wrap_mmap buffer");
    if (readback != fbase)
        die("gpu-selftest: host_ptr (%p) != wrapped base (%p) -- not zero-copy",
            (void *)readback, (void *)fbase);

    for (size_t i = 0; i < floats_per_page; i++) {
        if (readback[i] != (float)i)
            die("gpu-selftest: touched page mismatch at float %zu: got %g want %g",
                i, (double)readback[i], (double)i);
    }
    for (size_t i = floats_per_page; i < n_floats_total; i++) {
        if (readback[i] != (float)i)
            die("gpu-selftest: untouched page corrupted at float %zu: got %g want %g",
                i, (double)readback[i], (double)i);
    }
    fprintf(stderr, "gpu-selftest: wrap_mmap + sepia_touch dispatch verified (%zu pages)\n", n_pages);

    /* Alignment violation, reusing the same live mapping -- must fail
     * cleanly rather than crash. */
    SepiaGpuBuf *bad = sepia_gpu_wrap_mmap((char *)base + 1, page);
    if (bad) die("gpu-selftest: wrap_mmap(base+1) unexpectedly succeeded on a misaligned pointer");
    fprintf(stderr, "gpu-selftest: misaligned wrap_mmap correctly rejected\n");

    sepia_gpu_free(gbuf);
    munmap(base, total_len);

    /* alloc/free cycles: Shared vs Private, host_ptr NULL-ness. */
    SepiaGpuBuf *shared_buf = sepia_gpu_alloc(page, 0);
    if (!shared_buf) die("gpu-selftest: alloc(shared) failed");
    if (!sepia_gpu_host_ptr(shared_buf)) die("gpu-selftest: host_ptr(shared) unexpectedly NULL");
    sepia_gpu_free(shared_buf);

    SepiaGpuBuf *private_buf = sepia_gpu_alloc(page, 1);
    if (!private_buf) die("gpu-selftest: alloc(private) failed");
    if (sepia_gpu_host_ptr(private_buf) != NULL) die("gpu-selftest: host_ptr(private) expected NULL");
    sepia_gpu_free(private_buf);
    fprintf(stderr, "gpu-selftest: alloc/free (Shared + Private) verified\n");

    printf("gpu selftest ok\n");
}

/* ------------------------------ gpu compare tiny ---------------------------- */
/* --gpu-compare-tiny (local-only, needs --metal): the dev loop for Tasks
 * 4-8. Drives ONE real T=32 prefill through the UNMODIFIED tiny oracle (the
 * same model_load/model_forward_chunk the plain self-test uses) with
 * g_opcap set so the "op capture" call sites (see that section's doc
 * comment, above the attention section) snapshot real (input, params,
 * CPU-computed output) instances into the Cap*List globals; then replays
 * every captured instance through the matching sepia_gpu_* kernel and
 * reports the worst-case max-relative-error per op kind. Gate: every op
 * kind's worst instance must be <= SEPIA_GPU_COMPARE_TOL (2e-4 -- the P2
 * plan's Global Constraints (b); the CPU oracle accumulates reductions in
 * double, these kernels in f32, so a little drift is expected and is what
 * this tolerance is FOR, not a fudge factor for an actual bug). */

#define SEPIA_GPU_COMPARE_TOL 2e-4

static int g_gpu_compare_failed = 0;

/* `scale` is a captured INSTANCE's own dynamic range (max |expected value|
 * across that one (input,output) pair's output tensor), computed once per
 * instance below -- NOT pooled across every instance of the op kind. Pooling
 * across, e.g., all 448 matvec instances (wq/wo/router/experts families
 * alike) would let a large-magnitude family (e.g. router_w) loosen the floor
 * for a small-magnitude family (e.g. one expert's down-proj), which is
 * strictly looser than judging each instance against its own scale; per-
 * instance is what ggml's test-backend-ops does (each test case supplies its
 * own reference tensor's range), and this matches that convention exactly.
 * Plain |a-b|/max(|b|,eps) with a tiny FIXED eps still blows up near a
 * legitimate zero-crossing: e.g. one observed matvec instance (wo, a
 * cancellation-heavy dot product) has CPU=-1.31933007e-06,
 * GPU=-1.31875277e-06 -- an excellent 5.77e-10 absolute difference (float32
 * rounding noise at the scale of the ~64 summed terms) that a 1e-6 floor
 * reports as 4.4e-4 relative error, OVER the gate, purely because the true
 * value happens to sit near zero. Scaling the floor to a small fraction of
 * the op's own typical magnitude (rather than a fixed constant) is the
 * standard fix for this (ggml's test-backend-ops does the equivalent):
 * elements far below the instance's dynamic range are judged on absolute
 * closeness, not relative, while large-magnitude elements are still held to
 * the full relative tolerance. */
#define SEPIA_GPU_COMPARE_REL_FLOOR_FRAC 1e-3

static double gpu_compare_rel_err(float got, float want, double scale) {
    double diff = fabs((double)got - (double)want);
    double floor = fmax(1e-6, SEPIA_GPU_COMPARE_REL_FLOOR_FRAC * scale);
    double denom = fmax(fabs((double)want), floor);
    return diff / denom;
}

static void gpu_compare_report(const char *op, double max_rel, int n_instances) {
    printf("gpu-compare: %s max_rel_err %.3e (%d instances)\n", op, max_rel, n_instances);
    if (n_instances > 0 && max_rel > SEPIA_GPU_COMPARE_TOL) g_gpu_compare_failed = 1;
}

static void gpu_compare_rmsnorm(void) {
    int n_inst = g_cap_rmsnorm.count;
    if (n_inst == 0) {
        gpu_compare_report("rmsnorm", 0.0, 0);
        return;
    }
    SepiaGpuBuf **xb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **wb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **yb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: rmsnorm: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapRmsnorm *it = &g_cap_rmsnorm.items[i];
        xb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        wb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        yb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        if (!xb[i] || !wb[i] || !yb[i]) die("gpu-compare: rmsnorm: alloc failed");
        memcpy(sepia_gpu_host_ptr(xb[i]), it->x, sizeof(float) * (size_t)it->n);
        memcpy(sepia_gpu_host_ptr(wb[i]), it->w, sizeof(float) * (size_t)it->n);
        if (!sepia_gpu_rmsnorm(wb[i], xb[i], yb[i], 1, it->n, it->eps))
            die("gpu-compare: rmsnorm: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: rmsnorm: end failed");

    double max_rel = 0.0;
    for (int i = 0; i < n_inst; i++) {
        CapRmsnorm *it = &g_cap_rmsnorm.items[i];
        double scale = 0.0;
        for (int j = 0; j < it->n; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *y = (float *)sepia_gpu_host_ptr(yb[i]);
        for (int j = 0; j < it->n; j++) {
            double rel = gpu_compare_rel_err(y[j], it->y[j], scale);
            if (rel > max_rel) max_rel = rel;
        }
        sepia_gpu_free(xb[i]);
        sepia_gpu_free(wb[i]);
        sepia_gpu_free(yb[i]);
    }
    free(xb);
    free(wb);
    free(yb);
    gpu_compare_report("rmsnorm", max_rel, n_inst);
}

static void gpu_compare_matvec(void) {
    int n_inst = g_cap_matvec.count;
    if (n_inst == 0) {
        gpu_compare_report("matvec", 0.0, 0);
        return;
    }
    SepiaGpuBuf **wb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **xb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **yb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: matvec: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapMatvec *it = &g_cap_matvec.items[i];
        size_t wn = (size_t)it->out_dim * (size_t)it->in_dim;
        wb[i] = sepia_gpu_alloc(sizeof(float) * wn, 0);
        xb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->in_dim, 0);
        yb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->out_dim, 0);
        if (!wb[i] || !xb[i] || !yb[i]) die("gpu-compare: matvec: alloc failed");
        memcpy(sepia_gpu_host_ptr(wb[i]), it->w, sizeof(float) * wn);
        memcpy(sepia_gpu_host_ptr(xb[i]), it->x, sizeof(float) * (size_t)it->in_dim);
        if (!sepia_gpu_matvec(wb[i], xb[i], yb[i], it->out_dim, it->in_dim))
            die("gpu-compare: matvec: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: matvec: end failed");

    double max_rel = 0.0;
    int worst_i = -1, worst_j = -1;
    for (int i = 0; i < n_inst; i++) {
        CapMatvec *it = &g_cap_matvec.items[i];
        double scale = 0.0;
        for (int j = 0; j < it->out_dim; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *y = (float *)sepia_gpu_host_ptr(yb[i]);
        for (int j = 0; j < it->out_dim; j++) {
            double rel = gpu_compare_rel_err(y[j], it->y[j], scale);
            if (rel > max_rel) { max_rel = rel; worst_i = i; worst_j = j; }
        }
    }
    if (getenv("SEPIA_GPU_COMPARE_DEBUG") && worst_i >= 0) {
        CapMatvec *it = &g_cap_matvec.items[worst_i];
        float *y = (float *)sepia_gpu_host_ptr(yb[worst_i]);
        fprintf(stderr, "matvec debug: worst instance %d (out_dim=%d in_dim=%d) at j=%d: gpu=%.9g cpu=%.9g absdiff=%.3e\n",
                worst_i, it->out_dim, it->in_dim, worst_j, (double)y[worst_j], (double)it->y[worst_j],
                fabs((double)y[worst_j] - (double)it->y[worst_j]));
    }
    for (int i = 0; i < n_inst; i++) {
        sepia_gpu_free(wb[i]);
        sepia_gpu_free(xb[i]);
        sepia_gpu_free(yb[i]);
    }
    free(wb);
    free(xb);
    free(yb);
    gpu_compare_report("matvec", max_rel, n_inst);
}

static void gpu_compare_silu_mul(void) {
    int n_inst = g_cap_silu.count;
    if (n_inst == 0) {
        gpu_compare_report("silu_mul", 0.0, 0);
        return;
    }
    SepiaGpuBuf **gb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **ub = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **zb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: silu_mul: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapSiluMul *it = &g_cap_silu.items[i];
        gb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        ub[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        zb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        if (!gb[i] || !ub[i] || !zb[i]) die("gpu-compare: silu_mul: alloc failed");
        memcpy(sepia_gpu_host_ptr(gb[i]), it->g, sizeof(float) * (size_t)it->n);
        memcpy(sepia_gpu_host_ptr(ub[i]), it->u, sizeof(float) * (size_t)it->n);
        if (!sepia_gpu_silu_mul(gb[i], ub[i], zb[i], it->n))
            die("gpu-compare: silu_mul: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: silu_mul: end failed");

    double max_rel = 0.0;
    for (int i = 0; i < n_inst; i++) {
        CapSiluMul *it = &g_cap_silu.items[i];
        double scale = 0.0;
        for (int j = 0; j < it->n; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *z = (float *)sepia_gpu_host_ptr(zb[i]);
        for (int j = 0; j < it->n; j++) {
            double rel = gpu_compare_rel_err(z[j], it->y[j], scale);
            if (rel > max_rel) max_rel = rel;
        }
        sepia_gpu_free(gb[i]);
        sepia_gpu_free(ub[i]);
        sepia_gpu_free(zb[i]);
    }
    free(gb);
    free(ub);
    free(zb);
    gpu_compare_report("silu_mul", max_rel, n_inst);
}

static void gpu_compare_add(void) {
    int n_inst = g_cap_add.count;
    if (n_inst == 0) {
        gpu_compare_report("add", 0.0, 0);
        return;
    }
    SepiaGpuBuf **ab = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **bb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **ob = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: add: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapAdd *it = &g_cap_add.items[i];
        ab[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        bb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        ob[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        if (!ab[i] || !bb[i] || !ob[i]) die("gpu-compare: add: alloc failed");
        memcpy(sepia_gpu_host_ptr(ab[i]), it->a, sizeof(float) * (size_t)it->n);
        memcpy(sepia_gpu_host_ptr(bb[i]), it->b, sizeof(float) * (size_t)it->n);
        if (!sepia_gpu_add(ab[i], bb[i], ob[i], it->n))
            die("gpu-compare: add: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: add: end failed");

    double max_rel = 0.0;
    for (int i = 0; i < n_inst; i++) {
        CapAdd *it = &g_cap_add.items[i];
        double scale = 0.0;
        for (int j = 0; j < it->n; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *o = (float *)sepia_gpu_host_ptr(ob[i]);
        for (int j = 0; j < it->n; j++) {
            double rel = gpu_compare_rel_err(o[j], it->y[j], scale);
            if (rel > max_rel) max_rel = rel;
        }
        sepia_gpu_free(ab[i]);
        sepia_gpu_free(bb[i]);
        sepia_gpu_free(ob[i]);
    }
    free(ab);
    free(bb);
    free(ob);
    gpu_compare_report("add", max_rel, n_inst);
}

static void gpu_compare_softmax(void) {
    int n_inst = g_cap_softmax.count;
    if (n_inst == 0) {
        gpu_compare_report("softmax", 0.0, 0);
        return;
    }
    SepiaGpuBuf **xb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **yb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: softmax: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapSoftmax *it = &g_cap_softmax.items[i];
        xb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        yb[i] = sepia_gpu_alloc(sizeof(float) * (size_t)it->n, 0);
        if (!xb[i] || !yb[i]) die("gpu-compare: softmax: alloc failed");
        memcpy(sepia_gpu_host_ptr(xb[i]), it->scores, sizeof(float) * (size_t)it->n);
        if (!sepia_gpu_softmax(xb[i], yb[i], 1, it->n))
            die("gpu-compare: softmax: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: softmax: end failed");

    double max_rel = 0.0;
    for (int i = 0; i < n_inst; i++) {
        CapSoftmax *it = &g_cap_softmax.items[i];
        double scale = 0.0;
        for (int j = 0; j < it->n; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *y = (float *)sepia_gpu_host_ptr(yb[i]);
        for (int j = 0; j < it->n; j++) {
            double rel = gpu_compare_rel_err(y[j], it->y[j], scale);
            if (rel > max_rel) max_rel = rel;
        }
        sepia_gpu_free(xb[i]);
        sepia_gpu_free(yb[i]);
    }
    free(xb);
    free(yb);
    gpu_compare_report("softmax", max_rel, n_inst);
}

static void gpu_compare_sconv(void) {
    int n_inst = g_cap_sconv.count;
    if (n_inst == 0) {
        gpu_compare_report("sconv", 0.0, 0);
        return;
    }
    SepiaGpuBuf **wb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **hb = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **ib = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));
    SepiaGpuBuf **ob = xcalloc((size_t)n_inst, sizeof(SepiaGpuBuf *));

    if (!sepia_gpu_begin()) die("gpu-compare: sconv: begin failed");
    for (int i = 0; i < n_inst; i++) {
        CapSconv *it = &g_cap_sconv.items[i];
        size_t wn = (size_t)it->C * (size_t)it->K;
        size_t hn = (size_t)(it->K - 1) * (size_t)it->C;
        size_t tn = (size_t)it->T * (size_t)it->C;
        wb[i] = sepia_gpu_alloc(sizeof(float) * wn, 0);
        hb[i] = sepia_gpu_alloc(sizeof(float) * hn, 0);
        ib[i] = sepia_gpu_alloc(sizeof(float) * tn, 0);
        ob[i] = sepia_gpu_alloc(sizeof(float) * tn, 0);
        if (!wb[i] || !hb[i] || !ib[i] || !ob[i]) die("gpu-compare: sconv: alloc failed");
        memcpy(sepia_gpu_host_ptr(wb[i]), it->w, sizeof(float) * wn);
        memcpy(sepia_gpu_host_ptr(hb[i]), it->hist, sizeof(float) * hn);
        memcpy(sepia_gpu_host_ptr(ib[i]), it->in, sizeof(float) * tn);
        if (!sepia_gpu_sconv(wb[i], hb[i], ib[i], ob[i], it->C, it->K, it->T))
            die("gpu-compare: sconv: dispatch failed (instance %d)", i);
    }
    if (!sepia_gpu_end()) die("gpu-compare: sconv: end failed");

    double max_rel = 0.0;
    for (int i = 0; i < n_inst; i++) {
        CapSconv *it = &g_cap_sconv.items[i];
        size_t tn = (size_t)it->T * (size_t)it->C;
        double scale = 0.0;
        for (size_t j = 0; j < tn; j++) scale = fmax(scale, fabs((double)it->y[j]));
        float *o = (float *)sepia_gpu_host_ptr(ob[i]);
        for (size_t j = 0; j < tn; j++) {
            double rel = gpu_compare_rel_err(o[j], it->y[j], scale);
            if (rel > max_rel) max_rel = rel;
        }
        sepia_gpu_free(wb[i]);
        sepia_gpu_free(hb[i]);
        sepia_gpu_free(ib[i]);
        sepia_gpu_free(ob[i]);
    }
    free(wb);
    free(hb);
    free(ib);
    free(ob);
    gpu_compare_report("sconv", max_rel, n_inst);
}

/* ==================== Task 7: banded attention comparison ================= */
/* Shared GPU-replay-and-compare helper for BOTH Task 7 gates:
 *   Gate A (--gpu-compare-attn, run_gpu_compare_attn below): synthetic
 *     geometries, the REAL attn_forward_chunk as oracle.
 *   Gate B (folded into --gpu-compare-tiny via g_cap_attn, see cap_push_attn
 *     and CapAttn's doc comment): real tiny-oracle forward-pass instances.
 * Both need the exact same thing: given everything attn_forward_chunk
 * consumed for ONE call (weights via cfg/lw, the block's input x_normed,
 * the pre-call cache/history state, and its real output), replay the WHOLE
 * attention block via GPU dispatches -- wq/wk/wv/wr/wo matvec and k/v sconv
 * and q/k per-head rmsnorm (Task 3's already-gated kernels) plus rel-project
 * and banded attention (Task 7's new kernels) -- and report the max
 * relative error against the real output. This function contains ZERO
 * attention math of its own (no dot products, no softmax, no rel-logits
 * formula) -- every numeric step here is a dispatch into an existing
 * sepia_gpu_* kernel; the only host-side arithmetic is index/offset
 * bookkeeping (buffer sizes, cache concatenation) and the tau/kv_lo/kv_hi
 * per-token scalars, which the Task 7 plan explicitly assigns to the host
 * (see src/sepia_gpu.h's banded-attention doc). Buffers are uploaded fully
 * before any dispatch and read back only after sepia_gpu_end() -- never
 * reused across a write in between -- because Metal's batched encoder does
 * not wait between dispatches; overwriting a Shared buffer's host-side
 * contents while a prior dispatch that reads it is still just-encoded (not
 * yet executed) would be a race. */

static SepiaGpuBuf *gpu_upload_f32(const float *host, int64_t n) {
    SepiaGpuBuf *b = sepia_gpu_alloc(sizeof(float) * (size_t)(n > 0 ? n : 1), 0);
    if (!b) die("gpu-attn: alloc failed (n=%lld)", (long long)n);
    if (n > 0) memcpy(sepia_gpu_host_ptr(b), host, sizeof(float) * (size_t)n);
    return b;
}

/* Prints one per-case error-table row and returns this instance's
 * max-relative-error (per-instance-scaled against out_cpu's own dynamic
 * range, same convention as every other --gpu-compare-* gate in this file).
 * Does not itself gate/die -- callers accumulate the worst case and decide. */
static double gpu_attn_verify(const char *label, const Config *cfg, const LayerWeights *lw,
                               int T, int start_pos,
                               const float *x_normed,
                               const float *k_pre, const float *v_pre,
                               const float *k_hist_pre, const float *v_hist_pre,
                               const float *out_cpu) {
    int hidden = cfg->hidden_size;
    int H = lw->num_heads, Hkv = lw->num_kv_heads, Dh = lw->head_dim;
    int q_dim = lw->q_dim, kv_dim = lw->kv_dim, r_dim = lw->r_dim;
    int d_rel = cfg->d_rel, rel_extent = lw->rel_extent;
    int K = cfg->conv_kernel_size;
    int64_t cap = (int64_t)start_pos + (int64_t)T;

    /* ---- 1. projections: wq/wk/wv/wr matvec, one dispatch per (weight,t) --
     * every input buffer is uploaded before any dispatch, every output read
     * back only after sepia_gpu_end() (see the header note above). */
    float *q_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    float *k_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *r_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)r_dim);
    {
        SepiaGpuBuf *wq_buf = gpu_upload_f32(lw->wq, (int64_t)q_dim * hidden);
        SepiaGpuBuf *wk_buf = gpu_upload_f32(lw->wk, (int64_t)kv_dim * hidden);
        SepiaGpuBuf *wv_buf = gpu_upload_f32(lw->wv, (int64_t)kv_dim * hidden);
        SepiaGpuBuf *wr_buf = gpu_upload_f32(lw->wr, (int64_t)r_dim * hidden);
        SepiaGpuBuf **xb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));
        SepiaGpuBuf **yqb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));
        SepiaGpuBuf **ykb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));
        SepiaGpuBuf **yvb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));
        SepiaGpuBuf **yrb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (projections)", label);
        for (int t = 0; t < T; t++) {
            xb[t] = gpu_upload_f32(x_normed + (size_t)t * hidden, hidden);
            yqb[t] = sepia_gpu_alloc(sizeof(float) * (size_t)q_dim, 0);
            ykb[t] = sepia_gpu_alloc(sizeof(float) * (size_t)kv_dim, 0);
            yvb[t] = sepia_gpu_alloc(sizeof(float) * (size_t)kv_dim, 0);
            yrb[t] = sepia_gpu_alloc(sizeof(float) * (size_t)r_dim, 0);
            if (!yqb[t] || !ykb[t] || !yvb[t] || !yrb[t]) die("gpu-attn: %s: alloc failed (projections)", label);
            if (!sepia_gpu_matvec(wq_buf, xb[t], yqb[t], q_dim, hidden)) die("gpu-attn: %s: wq dispatch failed", label);
            if (!sepia_gpu_matvec(wk_buf, xb[t], ykb[t], kv_dim, hidden)) die("gpu-attn: %s: wk dispatch failed", label);
            if (!sepia_gpu_matvec(wv_buf, xb[t], yvb[t], kv_dim, hidden)) die("gpu-attn: %s: wv dispatch failed", label);
            if (!sepia_gpu_matvec(wr_buf, xb[t], yrb[t], r_dim, hidden)) die("gpu-attn: %s: wr dispatch failed", label);
        }
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (projections)", label);

        for (int t = 0; t < T; t++) {
            memcpy(q_raw + (size_t)t * q_dim, sepia_gpu_host_ptr(yqb[t]), sizeof(float) * (size_t)q_dim);
            memcpy(k_raw + (size_t)t * kv_dim, sepia_gpu_host_ptr(ykb[t]), sizeof(float) * (size_t)kv_dim);
            memcpy(v_raw + (size_t)t * kv_dim, sepia_gpu_host_ptr(yvb[t]), sizeof(float) * (size_t)kv_dim);
            memcpy(r_raw + (size_t)t * r_dim, sepia_gpu_host_ptr(yrb[t]), sizeof(float) * (size_t)r_dim);
            sepia_gpu_free(xb[t]);
            sepia_gpu_free(yqb[t]);
            sepia_gpu_free(ykb[t]);
            sepia_gpu_free(yvb[t]);
            sepia_gpu_free(yrb[t]);
        }
        free(xb);
        free(yqb);
        free(ykb);
        free(yvb);
        free(yrb);
        sepia_gpu_free(wq_buf);
        sepia_gpu_free(wk_buf);
        sepia_gpu_free(wv_buf);
        sepia_gpu_free(wr_buf);
    }

    /* ---- 2. sconv: k_raw/v_raw + pre-call history -> k_sconv/v_sconv ---- */
    float *k_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    {
        SepiaGpuBuf *kw_buf = gpu_upload_f32(lw->k_sconv_w, (int64_t)kv_dim * K);
        SepiaGpuBuf *vw_buf = gpu_upload_f32(lw->v_sconv_w, (int64_t)kv_dim * K);
        SepiaGpuBuf *kh_buf = gpu_upload_f32(k_hist_pre, (int64_t)(K - 1) * kv_dim);
        SepiaGpuBuf *vh_buf = gpu_upload_f32(v_hist_pre, (int64_t)(K - 1) * kv_dim);
        SepiaGpuBuf *ki_buf = gpu_upload_f32(k_raw, (int64_t)T * kv_dim);
        SepiaGpuBuf *vi_buf = gpu_upload_f32(v_raw, (int64_t)T * kv_dim);
        SepiaGpuBuf *ko_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)kv_dim, 0);
        SepiaGpuBuf *vo_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)kv_dim, 0);
        if (!ko_buf || !vo_buf) die("gpu-attn: %s: alloc failed (sconv)", label);

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (sconv)", label);
        if (!sepia_gpu_sconv(kw_buf, kh_buf, ki_buf, ko_buf, kv_dim, K, T)) die("gpu-attn: %s: k sconv dispatch failed", label);
        if (!sepia_gpu_sconv(vw_buf, vh_buf, vi_buf, vo_buf, kv_dim, K, T)) die("gpu-attn: %s: v sconv dispatch failed", label);
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (sconv)", label);

        memcpy(k_sconv, sepia_gpu_host_ptr(ko_buf), sizeof(float) * (size_t)T * (size_t)kv_dim);
        memcpy(v_sconv, sepia_gpu_host_ptr(vo_buf), sizeof(float) * (size_t)T * (size_t)kv_dim);
        sepia_gpu_free(kw_buf);
        sepia_gpu_free(vw_buf);
        sepia_gpu_free(kh_buf);
        sepia_gpu_free(vh_buf);
        sepia_gpu_free(ki_buf);
        sepia_gpu_free(vi_buf);
        sepia_gpu_free(ko_buf);
        sepia_gpu_free(vo_buf);
    }

    /* ---- 3. per-head rmsnorm: q_norm/k_norm broadcast across every (t,h)
     * slice -- q_raw's [T,q_dim] layout is already exactly [T*H,Dh] rows
     * (q_dim=H*Dh), so ONE sepia_gpu_rmsnorm call covers the whole tensor
     * (same trick for k_sconv's [T*Hkv,Dh] rows). No norm on v (sec.2). ---- */
    float *q_normed = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    float *k_normed = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    {
        SepiaGpuBuf *qn_buf = gpu_upload_f32(lw->q_norm, Dh);
        SepiaGpuBuf *qx_buf = gpu_upload_f32(q_raw, (int64_t)T * q_dim);
        SepiaGpuBuf *qy_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)q_dim, 0);
        SepiaGpuBuf *kn_buf = gpu_upload_f32(lw->k_norm, Dh);
        SepiaGpuBuf *kx_buf = gpu_upload_f32(k_sconv, (int64_t)T * kv_dim);
        SepiaGpuBuf *ky_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)kv_dim, 0);
        if (!qy_buf || !ky_buf) die("gpu-attn: %s: alloc failed (rmsnorm)", label);

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (rmsnorm)", label);
        if (!sepia_gpu_rmsnorm(qn_buf, qx_buf, qy_buf, (int64_t)T * H, Dh, (float)cfg->rms_norm_eps))
            die("gpu-attn: %s: q_norm dispatch failed", label);
        if (!sepia_gpu_rmsnorm(kn_buf, kx_buf, ky_buf, (int64_t)T * Hkv, Dh, (float)cfg->rms_norm_eps))
            die("gpu-attn: %s: k_norm dispatch failed", label);
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (rmsnorm)", label);

        memcpy(q_normed, sepia_gpu_host_ptr(qy_buf), sizeof(float) * (size_t)T * (size_t)q_dim);
        memcpy(k_normed, sepia_gpu_host_ptr(ky_buf), sizeof(float) * (size_t)T * (size_t)kv_dim);
        sepia_gpu_free(qn_buf);
        sepia_gpu_free(qx_buf);
        sepia_gpu_free(qy_buf);
        sepia_gpu_free(kn_buf);
        sepia_gpu_free(kx_buf);
        sepia_gpu_free(ky_buf);
    }

    /* ---- 4. assemble the full cache (pre-call history ++ this call's new
     * rows) -- plain memcpy, no math; mirrors what attn_forward_chunk's own
     * lc->k/lc->v memcpy-write-back produces before its attention loop
     * reads them. ---- */
    float *k_full = xmalloc(sizeof(float) * (size_t)cap * (size_t)kv_dim);
    float *v_full = xmalloc(sizeof(float) * (size_t)cap * (size_t)kv_dim);
    memcpy(k_full, k_pre, sizeof(float) * (size_t)start_pos * (size_t)kv_dim);
    memcpy(k_full + (size_t)start_pos * (size_t)kv_dim, k_normed, sizeof(float) * (size_t)T * (size_t)kv_dim);
    memcpy(v_full, v_pre, sizeof(float) * (size_t)start_pos * (size_t)kv_dim);
    memcpy(v_full + (size_t)start_pos * (size_t)kv_dim, v_sconv, sizeof(float) * (size_t)T * (size_t)kv_dim);

    /* ---- 5. tau / kv_lo / kv_hi (host-computed per-token scalars, exactly
     * as the Task 7 plan assigns -- "kv_lo/hi per token computed host-
     * side") and q_scaled = q_normed * tau (the "host pre-applies tau to q"
     * design choice, src/sepia_gpu.h). This mirrors attn_forward_chunk's
     * own tau/kv_lo/kv_hi formula (src/sepia.c) -- a trivial, orthogonal
     * scalar computation, not the attention math this gate targets. ---- */
    int64_t *kv_lo = xmalloc(sizeof(int64_t) * (size_t)T);
    int64_t *kv_hi = xmalloc(sizeof(int64_t) * (size_t)T);
    float *tau = xmalloc(sizeof(float) * (size_t)T);
    int have_log_scaling = (!lw->is_sliding) && cfg->has_log_scaling_floor;
    for (int t = 0; t < T; t++) {
        int q_pos = start_pos + t;
        double tv = 1.0;
        if (have_log_scaling) {
            double effective_n = (double)(q_pos + 1);
            double ratio = effective_n / (double)cfg->log_scaling_n_floor;
            if (ratio < 1.0) ratio = 1.0;
            tv = 1.0 + cfg->log_scaling_alpha * log(ratio);
        }
        tau[t] = (float)tv;
        int64_t kvlo = lw->is_sliding ? (int64_t)q_pos - (int64_t)cfg->sliding_window_size + 1 : 0;
        if (kvlo < 0) kvlo = 0;
        kv_lo[t] = kvlo;
        kv_hi[t] = q_pos;
    }
    float *q_scaled = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < q_dim; i++)
            q_scaled[(size_t)t * q_dim + i] = q_normed[(size_t)t * q_dim + i] * tau[t];

    /* ---- 6. rel-project (Task 7 kernel 1) ---- */
    float *rel_logits = xmalloc(sizeof(float) * (size_t)T * (size_t)H * (size_t)rel_extent);
    {
        SepiaGpuBuf *rv_buf = gpu_upload_f32(r_raw, (int64_t)T * r_dim);
        SepiaGpuBuf *rp_buf = gpu_upload_f32(lw->rel_proj, (int64_t)d_rel * rel_extent);
        SepiaGpuBuf *rl_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)H * (size_t)rel_extent, 0);
        if (!rl_buf) die("gpu-attn: %s: alloc failed (rel_project)", label);

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (rel_project)", label);
        if (!sepia_gpu_rel_project(rv_buf, rp_buf, rl_buf, T, H, d_rel, rel_extent))
            die("gpu-attn: %s: rel_project dispatch failed", label);
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (rel_project)", label);

        memcpy(rel_logits, sepia_gpu_host_ptr(rl_buf), sizeof(float) * (size_t)T * (size_t)H * (size_t)rel_extent);
        sepia_gpu_free(rv_buf);
        sepia_gpu_free(rp_buf);
        sepia_gpu_free(rl_buf);
    }

    /* ---- 7. banded attention (Task 7 kernel 2) ---- */
    float *attn_concat = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    {
        SepiaGpuBuf *q_buf = gpu_upload_f32(q_scaled, (int64_t)T * q_dim);
        SepiaGpuBuf *k_buf = gpu_upload_f32(k_full, cap * kv_dim);
        SepiaGpuBuf *v_buf = gpu_upload_f32(v_full, cap * kv_dim);
        SepiaGpuBuf *rl_buf = gpu_upload_f32(rel_logits, (int64_t)T * H * rel_extent);
        SepiaGpuBuf *out_buf = sepia_gpu_alloc(sizeof(float) * (size_t)T * (size_t)q_dim, 0);
        if (!out_buf) die("gpu-attn: %s: alloc failed (banded_attn)", label);

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (banded_attn)", label);
        if (!sepia_gpu_banded_attn(q_buf, k_buf, v_buf, rl_buf, out_buf, kv_lo, kv_hi, tau,
                                    T, H, Hkv, Dh, rel_extent, start_pos, kv_dim, 1.0f / (float)Dh))
            die("gpu-attn: %s: banded_attn dispatch failed", label);
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (banded_attn)", label);

        memcpy(attn_concat, sepia_gpu_host_ptr(out_buf), sizeof(float) * (size_t)T * (size_t)q_dim);
        sepia_gpu_free(q_buf);
        sepia_gpu_free(k_buf);
        sepia_gpu_free(v_buf);
        sepia_gpu_free(rl_buf);
        sepia_gpu_free(out_buf);
    }

    /* ---- 8. wo matvec (Task 3's kernel, one dispatch per t) ---- */
    float *out_gpu = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    {
        SepiaGpuBuf *wo_buf = gpu_upload_f32(lw->wo, (int64_t)hidden * q_dim);
        SepiaGpuBuf **xb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));
        SepiaGpuBuf **yb = xcalloc((size_t)T, sizeof(SepiaGpuBuf *));

        if (!sepia_gpu_begin()) die("gpu-attn: %s: begin failed (wo)", label);
        for (int t = 0; t < T; t++) {
            xb[t] = gpu_upload_f32(attn_concat + (size_t)t * q_dim, q_dim);
            yb[t] = sepia_gpu_alloc(sizeof(float) * (size_t)hidden, 0);
            if (!yb[t]) die("gpu-attn: %s: alloc failed (wo)", label);
            if (!sepia_gpu_matvec(wo_buf, xb[t], yb[t], hidden, q_dim)) die("gpu-attn: %s: wo dispatch failed", label);
        }
        if (!sepia_gpu_end()) die("gpu-attn: %s: end failed (wo)", label);

        for (int t = 0; t < T; t++) {
            memcpy(out_gpu + (size_t)t * hidden, sepia_gpu_host_ptr(yb[t]), sizeof(float) * (size_t)hidden);
            sepia_gpu_free(xb[t]);
            sepia_gpu_free(yb[t]);
        }
        free(xb);
        free(yb);
        sepia_gpu_free(wo_buf);
    }

    /* ---- 9. compare against the real CPU output, per-instance-scaled ---- */
    size_t total = (size_t)T * (size_t)hidden;
    double scale = 0.0;
    for (size_t i = 0; i < total; i++) scale = fmax(scale, fabs((double)out_cpu[i]));
    double max_rel = 0.0;
    for (size_t i = 0; i < total; i++) {
        double rel = gpu_compare_rel_err(out_gpu[i], out_cpu[i], scale);
        if (rel > max_rel) max_rel = rel;
    }

    printf("gpu-compare-attn: %-40s T=%-3d n_kv[0..T-1]=%lld..%lld max_rel_err %.3e  %s\n",
           label, T, (long long)(kv_hi[0] - kv_lo[0] + 1), (long long)(kv_hi[T - 1] - kv_lo[T - 1] + 1),
           max_rel, max_rel <= SEPIA_GPU_COMPARE_TOL ? "ok" : "FAIL");

    free(q_raw);
    free(k_raw);
    free(v_raw);
    free(r_raw);
    free(k_sconv);
    free(v_sconv);
    free(q_normed);
    free(k_normed);
    free(k_full);
    free(v_full);
    free(kv_lo);
    free(kv_hi);
    free(tau);
    free(q_scaled);
    free(rel_logits);
    free(attn_concat);
    free(out_gpu);

    return max_rel;
}

/* --gpu-compare-attn (local-only, needs --metal): Task 7 Gate A. Builds a
 * standalone Config/LayerWeights/LayerCache at the two REAL per-layer-type
 * geometries (docs/inkling-config.json: sliding H64/Hkv16/Dh128/window512/
 * rel512=window; global H64/Hkv8/Dh128/rel1024, d_rel=16 both,
 * log_scaling_alpha=0.1/log_scaling_n_floor=128000 for the global case)
 * with randomized weights and cache history -- hidden_size is deliberately
 * shrunk to 64 here (vs the real 6144) since it only sizes the wq/wk/wv/wr/
 * wo *projection* matvecs, already Task 3's gated kernel and orthogonal to
 * the rel-project/banded-attention math this gate targets; every dimension
 * that DOES matter for that math (H, Hkv, Dh, rel_extent, d_rel, window,
 * log-scaling params) is the exact real value. For each (label, is_sliding,
 * q_pos, T) case, calls the REAL attn_forward_chunk (unmodified CPU oracle)
 * then gpu_attn_verify above; never reimplements attn_forward_chunk's math,
 * only constructs the synthetic structs it consumes (the Task 7 plan's
 * explicit requirement). */

static float *gpu_attn_randbuf(unsigned *seed, size_t n, float scale) {
    float *b = xmalloc(sizeof(float) * (n ? n : 1));
    for (size_t i = 0; i < n; i++)
        b[i] = (((float)rand_r(seed) / (float)RAND_MAX) * 2.0f - 1.0f) * scale;
    return b;
}

typedef struct {
    const char *label;
    int is_sliding;
    int q_pos;
    int T;
} AttnGateCase;

static void run_gpu_compare_attn(void) {
    if (!sepia_gpu_available()) die("gpu-compare-attn: requires --metal to have initialized the GPU runtime");

    enum { HIDDEN = 64, H = 64, HKV_SLIDING = 16, HKV_GLOBAL = 8, DH = 128, D_REL = 16,
           WINDOW = 512, REL_EXTENT_GLOBAL = 1024, K = 4 };

    Config cfg = {0};
    cfg.hidden_size = HIDDEN;
    cfg.d_rel = D_REL;
    cfg.conv_kernel_size = K;
    cfg.rms_norm_eps = 1e-6;
    cfg.sliding_window_size = WINDOW;
    cfg.rel_extent = REL_EXTENT_GLOBAL;
    cfg.has_log_scaling_floor = 1;
    cfg.log_scaling_n_floor = 128000;
    cfg.log_scaling_alpha = 0.1;

    /* n_kv > rel_extent only ever happens for global layers (sliding's
     * kv_lo grows with q_pos, so n_kv never exceeds window==rel_extent
     * there) -- so the ">" and "q_pos>log_floor" cases are global-only,
     * matching the plan's own parenthetical ("global long-context:
     * positions beyond the band get content-only scores"). The multi-token
     * case below is global for the same reason, and is placed so the SAME
     * T=8 dispatch straddles the rel_extent=1024 boundary: with q_pos=1027
     * (start_pos=1020), kv_lo=0 always, so per-token max distance is just
     * q_pos itself -- tokens q_pos 1020..1023 (t=0..3) stay under 1024 and
     * never hit the band cutoff, while q_pos 1024..1027 (t=4..7) do, so the
     * cutoff toggles mid-chunk within one dispatch (unlike the single-token
     * "> rel_extent" case above, which only ever exercises it in isolation). */
    AttnGateCase cases[] = {
        { "sliding n_kv < rel_extent",                  1, 100,               1 },
        { "sliding n_kv == rel_extent",                 1, WINDOW - 1,        1 },
        { "global  n_kv < rel_extent",                  0, 100,               1 },
        { "global  n_kv == rel_extent",                 0, REL_EXTENT_GLOBAL - 1, 1 },
        { "global  n_kv > rel_extent (band cutoff live)", 0, 5000,            1 },
        { "global  q_pos > log_scaling_n_floor (tau!=1)", 0, 130000,          1 },
        { "global  multi-token chunk (band edge mid-chunk)", 0, 1027,         8 },
    };
    int n_cases = (int)(sizeof(cases) / sizeof(cases[0]));

    double worst = 0.0;
    const char *worst_label = NULL;
    for (int c = 0; c < n_cases; c++) {
        AttnGateCase *tc = &cases[c];
        int T = tc->T;
        int start_pos = tc->q_pos - T + 1;
        if (start_pos < 0) die("gpu-compare-attn: case '%s': q_pos-T+1 < 0", tc->label);

        LayerWeights lw = {0};
        lw.is_sliding = tc->is_sliding;
        lw.num_heads = H;
        lw.num_kv_heads = tc->is_sliding ? HKV_SLIDING : HKV_GLOBAL;
        lw.head_dim = DH;
        lw.rel_extent = tc->is_sliding ? WINDOW : REL_EXTENT_GLOBAL;
        lw.q_dim = lw.num_heads * lw.head_dim;
        lw.kv_dim = lw.num_kv_heads * lw.head_dim;
        lw.r_dim = lw.num_heads * D_REL;

        unsigned seed = 20260720u ^ (unsigned)c;
        float *wq = gpu_attn_randbuf(&seed, (size_t)lw.q_dim * HIDDEN, 0.05f);
        float *wk = gpu_attn_randbuf(&seed, (size_t)lw.kv_dim * HIDDEN, 0.05f);
        float *wv = gpu_attn_randbuf(&seed, (size_t)lw.kv_dim * HIDDEN, 0.05f);
        float *wr = gpu_attn_randbuf(&seed, (size_t)lw.r_dim * HIDDEN, 0.05f);
        float *wo = gpu_attn_randbuf(&seed, (size_t)HIDDEN * lw.q_dim, 0.05f);
        float *q_norm = gpu_attn_randbuf(&seed, (size_t)DH, 1.0f);
        float *k_norm = gpu_attn_randbuf(&seed, (size_t)DH, 1.0f);
        float *rel_proj = gpu_attn_randbuf(&seed, (size_t)D_REL * lw.rel_extent, 0.05f);
        float *k_sconv_w = gpu_attn_randbuf(&seed, (size_t)lw.kv_dim * K, 0.1f);
        float *v_sconv_w = gpu_attn_randbuf(&seed, (size_t)lw.kv_dim * K, 0.1f);
        lw.wq = wq; lw.wk = wk; lw.wv = wv; lw.wr = wr; lw.wo = wo;
        lw.q_norm = q_norm; lw.k_norm = k_norm; lw.rel_proj = rel_proj;
        lw.k_sconv_w = k_sconv_w; lw.v_sconv_w = v_sconv_w;

        float *x_normed = gpu_attn_randbuf(&seed, (size_t)T * HIDDEN, 0.2f);

        LayerCache lc = {0};
        lc.k = gpu_attn_randbuf(&seed, (size_t)(start_pos + T) * (size_t)lw.kv_dim, 0.2f);
        lc.v = gpu_attn_randbuf(&seed, (size_t)(start_pos + T) * (size_t)lw.kv_dim, 0.2f);
        lc.k_hist = gpu_attn_randbuf(&seed, (size_t)(K - 1) * (size_t)lw.kv_dim, 0.2f);
        lc.v_hist = gpu_attn_randbuf(&seed, (size_t)(K - 1) * (size_t)lw.kv_dim, 0.2f);
        lc.len = start_pos;

        /* Snapshot BEFORE calling attn_forward_chunk, which mutates lc->k/
         * lc->v/lc->k_hist/lc->v_hist in place. */
        float *k_pre = fdup(lc.k, (size_t)start_pos * (size_t)lw.kv_dim);
        float *v_pre = fdup(lc.v, (size_t)start_pos * (size_t)lw.kv_dim);
        float *k_hist_pre = fdup(lc.k_hist, (size_t)(K - 1) * (size_t)lw.kv_dim);
        float *v_hist_pre = fdup(lc.v_hist, (size_t)(K - 1) * (size_t)lw.kv_dim);

        float *out_cpu = xmalloc(sizeof(float) * (size_t)T * HIDDEN);
        attn_forward_chunk(&cfg, &lw, &lc, x_normed, T, start_pos, out_cpu);

        double max_rel = gpu_attn_verify(tc->label, &cfg, &lw, T, start_pos, x_normed,
                                          k_pre, v_pre, k_hist_pre, v_hist_pre, out_cpu);
        if (max_rel > worst) { worst = max_rel; worst_label = tc->label; }

        free(wq); free(wk); free(wv); free(wr); free(wo);
        free(q_norm); free(k_norm); free(rel_proj); free(k_sconv_w); free(v_sconv_w);
        free(x_normed);
        free(lc.k); free(lc.v); free(lc.k_hist); free(lc.v_hist);
        free(k_pre); free(v_pre); free(k_hist_pre); free(v_hist_pre);
        free(out_cpu);
    }

    printf("gpu-compare-attn: worst case '%s' max_rel_err %.3e (tolerance %.0e)\n",
           worst_label ? worst_label : "(none)", worst, SEPIA_GPU_COMPARE_TOL);
    if (worst > SEPIA_GPU_COMPARE_TOL)
        die("gpu-compare-attn: at least one case exceeded the %.0e relative-error tolerance", SEPIA_GPU_COMPARE_TOL);
    printf("gpu compare attn ok\n");
}

static void run_gpu_compare_tiny(void) {
    if (!sepia_gpu_available()) die("gpu-compare-tiny: requires --metal to have initialized the GPU runtime");

    const char *config_path = env_or("SEPIA_CONFIG_PATH", CONFIG_PATH);
    const char *weights_path = env_or("SEPIA_WEIGHTS_PATH", WEIGHTS_PATH);
    const char *ref_path = env_or("SEPIA_REF_PATH", REF_PATH);

    Model m = model_load(config_path, weights_path);
    OracleRef ref = load_oracle_ref(ref_path);
    int full_len = ref.full_ids.len; /* the tiny oracle's T=32 teacher-forcing prefill */
    int unpadded = m.cfg.unpadded_vocab_size;

    /* +1 of capacity: the T=32 prefill below fills positions [0,full_len),
     * then one incremental T=1 decode step (finding 2's fix) continues the
     * SAME cache at start_pos=full_len, writing k/v for that one extra
     * position -- mirrors run_self_test's prefill-then-decode shape. */
    Cache *c = cache_create(&m, full_len + 1);
    float *logits = xmalloc(sizeof(float) * (size_t)full_len * (size_t)unpadded);

    g_opcap = 1;
    model_forward_chunk(&m, c, ref.full_ids.ids, full_len, 0, logits, NULL);

    /* The prefill above is the cache's first-ever chunk, so every sconv
     * hist buffer it captures is all-zero (cache_create zero-inits them) --
     * that branch of sepia_gpu_sconv was never exercised with real history.
     * Continue the SAME cache with one incremental decode step (T=1,
     * start_pos=full_len, true greedy: feed the prefill's own last-position
     * prediction forward, same convention as run_self_test's decode loop)
     * so the sconv instances captured from THIS step see the nonzero
     * history the prefill just built up. */
    int next_id = argmax_f(logits + (size_t)(full_len - 1) * unpadded, unpadded);
    float *decode_logits = xmalloc(sizeof(float) * (size_t)unpadded);
    model_forward_chunk(&m, c, &next_id, 1, full_len, decode_logits, NULL);
    free(decode_logits);
    g_opcap = 0;

    free(logits);
    cache_free(c);

    fprintf(stderr, "gpu-compare-tiny: captured %d rmsnorm, %d matvec, %d silu_mul, %d add, "
                     "%d softmax, %d sconv, %d attn instances from a %d-token prefill + 1-token decode step\n",
            g_cap_rmsnorm.count, g_cap_matvec.count, g_cap_silu.count, g_cap_add.count,
            g_cap_softmax.count, g_cap_sconv.count, g_cap_attn.count, full_len);

    /* Regression guard for the history coverage this fix adds: if every
     * captured sconv instance ever went back to all-zero hist (e.g. a
     * future edit stops running the decode step above), fail loudly here
     * rather than silently losing that branch's coverage again. */
    int sconv_hist_nonzero = 0;
    for (int i = 0; i < g_cap_sconv.count && !sconv_hist_nonzero; i++) {
        CapSconv *it = &g_cap_sconv.items[i];
        size_t hn = (size_t)(it->K - 1) * (size_t)it->C;
        for (size_t j = 0; j < hn; j++) {
            if (it->hist[j] != 0.0f) { sconv_hist_nonzero = 1; break; }
        }
    }
    if (g_cap_sconv.count > 0 && !sconv_hist_nonzero)
        die("gpu-compare-tiny: no captured sconv instance has nonzero history "
            "(decode-step coverage regressed)");

    gpu_compare_rmsnorm();
    gpu_compare_matvec();
    gpu_compare_silu_mul();
    gpu_compare_add();
    gpu_compare_softmax();
    gpu_compare_sconv();

    /* Task 7 Gate B: attention-swap comparison -- for each captured
     * attn_forward_chunk instance (layer 0 / last layer, from the prefill
     * and the decode step, restricted by g_opcap_selected_layer same as
     * every other Cap* site), replay the whole attention block via GPU
     * dispatches (gpu_attn_verify, shared with Gate A above) and compare
     * against the real per-layer output at SEPIA_GPU_COMPARE_TOL. */
    double attn_worst = 0.0;
    for (int i = 0; i < g_cap_attn.count; i++) {
        CapAttn *it = &g_cap_attn.items[i];
        char label[64];
        snprintf(label, sizeof label, "tiny instance %d (T=%d start_pos=%d)", i, it->T, it->start_pos);
        double rel = gpu_attn_verify(label, it->cfg, it->lw, it->T, it->start_pos, it->x_normed,
                                      it->k_pre, it->v_pre, it->k_hist_pre, it->v_hist_pre, it->out);
        if (rel > attn_worst) attn_worst = rel;
    }
    gpu_compare_report("attn (Gate B attention-swap)", attn_worst, g_cap_attn.count);

    if (g_gpu_compare_failed)
        die("gpu-compare-tiny: at least one op exceeded the %.0e relative-error tolerance", SEPIA_GPU_COMPARE_TOL);
    printf("gpu compare ok\n");
}

/* ==================== Task 8: full tiny-model GPU forward ================= */
/* The composition milestone: every kernel from Tasks 3-7 wired into a
 * complete per-layer graph (model_forward_chunk_gpu, forward-declared above
 * model_forward_chunk) that must reproduce the tiny CPU oracle EXACTLY
 * (--metal's plain self-test path, token-exact prefill 32/32 + decode
 * 20/20 -- see .superpowers/sdd/task-8-report.md for the bring-up notes and
 * the sconv-history/MoE-readback/w13 design writeups summarized inline
 * below). Every helper here is a thin, self-contained (own begin/flush/end)
 * wrapper around one or two sepia_gpu_* dispatches plus the upload/readback
 * around them -- "per-layer sync is acceptable at tiny scale" (the plan's
 * own words) is taken literally: every op gets its own round trip, since
 * correctness (not throughput) is this task's gate. Real-model performance
 * (batched encoding across many dispatches per sepia_gpu_begin/end) is
 * Task 9+'s concern. */

static void gpu_matvec_one(const float *w, const float *x, float *y, int64_t out_dim, int64_t in_dim) {
    SepiaGpuBuf *wb = gpu_upload_f32(w, out_dim * in_dim);
    SepiaGpuBuf *xb = gpu_upload_f32(x, in_dim);
    SepiaGpuBuf *yb = sepia_gpu_alloc(sizeof(float) * (size_t)out_dim, 0);
    if (!yb) die("gpu-forward: matvec: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: matvec: begin failed");
    if (!sepia_gpu_matvec(wb, xb, yb, out_dim, in_dim)) die("gpu-forward: matvec: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: matvec: end failed");
    memcpy(y, sepia_gpu_host_ptr(yb), sizeof(float) * (size_t)out_dim);
    sepia_gpu_free(wb);
    sepia_gpu_free(xb);
    sepia_gpu_free(yb);
}

/* w[n] is broadcast across every one of `rows` [rows,n] rows -- the same
 * trick gpu_attn_verify already relies on for q_norm/k_norm (and here also
 * for embed_norm/attn_norm/mlp_norm/final_norm's single-row-shape case,
 * rows=T): one dispatch, not a per-row loop. */
static void gpu_rmsnorm_rows(const float *w, const float *x, float *y, int64_t rows, int64_t n, float eps) {
    SepiaGpuBuf *wb = gpu_upload_f32(w, n);
    SepiaGpuBuf *xb = gpu_upload_f32(x, rows * n);
    SepiaGpuBuf *yb = sepia_gpu_alloc(sizeof(float) * (size_t)(rows * n), 0);
    if (!yb) die("gpu-forward: rmsnorm: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: rmsnorm: begin failed");
    if (!sepia_gpu_rmsnorm(wb, xb, yb, rows, n, eps)) die("gpu-forward: rmsnorm: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: rmsnorm: end failed");
    memcpy(y, sepia_gpu_host_ptr(yb), sizeof(float) * (size_t)(rows * n));
    sepia_gpu_free(wb);
    sepia_gpu_free(xb);
    sepia_gpu_free(yb);
}

static void gpu_silu_mul_pair(const float *g, const float *u, float *z, int64_t n) {
    SepiaGpuBuf *gb = gpu_upload_f32(g, n);
    SepiaGpuBuf *ub = gpu_upload_f32(u, n);
    SepiaGpuBuf *zb = sepia_gpu_alloc(sizeof(float) * (size_t)n, 0);
    if (!zb) die("gpu-forward: silu_mul: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: silu_mul: begin failed");
    if (!sepia_gpu_silu_mul(gb, ub, zb, n)) die("gpu-forward: silu_mul: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: silu_mul: end failed");
    memcpy(z, sepia_gpu_host_ptr(zb), sizeof(float) * (size_t)n);
    sepia_gpu_free(gb);
    sepia_gpu_free(ub);
    sepia_gpu_free(zb);
}

/* x[i] += b[i], via the GPU add kernel (x is both the source and the
 * destination host array -- a fresh output buffer is read back into it,
 * never aliased at the SepiaGpuBuf level). */
static void gpu_add_into(float *x, const float *b, int64_t n) {
    SepiaGpuBuf *ab = gpu_upload_f32(x, n);
    SepiaGpuBuf *bb = gpu_upload_f32(b, n);
    SepiaGpuBuf *ob = sepia_gpu_alloc(sizeof(float) * (size_t)n, 0);
    if (!ob) die("gpu-forward: add: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: add: begin failed");
    if (!sepia_gpu_add(ab, bb, ob, n)) die("gpu-forward: add: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: add: end failed");
    memcpy(x, sepia_gpu_host_ptr(ob), sizeof(float) * (size_t)n);
    sepia_gpu_free(ab);
    sepia_gpu_free(bb);
    sepia_gpu_free(ob);
}

/* ---- sconv, with the Task 3 tracked gap closed: design choice (b) ----
 * The GPU sconv kernel (metal/ops.metal, Task 3) never writes an updated
 * history buffer back -- by design, to avoid a read/write hazard within one
 * dispatch (option (c) from the task brief). Task 8 needs decode (T=1 steps
 * chaining) and prefill (T=32, history rolling across the whole chunk) to
 * both keep `hist` correct across calls. Of the three options the plan
 * offered, this picks (b), host-side history update between steps: `hist`
 * (LayerCache's k_hist/v_hist/attn_hist/mlp_hist) is already a plain host
 * float array (never itself a GPU buffer -- gpu_upload_f32 copies it INTO a
 * transient one for the dispatch), and `in` is already host-resident too
 * (every caller here builds it from a prior GPU readback or a host gather).
 * The update itself is pure data movement with no floating-point operation
 * at all (just picking which already-computed values become the next
 * call's history), so it reproduces sconv_apply's own tail exactly (src/
 * sepia.c): new hist = the last K-1 rows of concat(old hist, in). Options
 * (a)/(c) (a GPU kernel doing the same roll) were rejected because they'd
 * spend a whole extra dispatch (plus the buffer plumbing for a second
 * output) on an operation that is provably not floating-point math -- at
 * tiny scale the extra host-side memmove/memcpy is unmeasurable, and this
 * keeps the GPU sconv kernel itself unchanged from Task 3. See
 * task-8-report.md's "sconv-history design" section for the full writeup. */
static void gpu_sconv_step(const float *w, float *hist /*[K-1,C] host, in/out*/,
                            const float *in, float *out, int64_t C, int64_t K, int64_t T) {
    int64_t Km1 = K - 1;
    SepiaGpuBuf *wb = gpu_upload_f32(w, C * K);
    SepiaGpuBuf *hb = gpu_upload_f32(hist, Km1 * C);
    SepiaGpuBuf *ib = gpu_upload_f32(in, T * C);
    SepiaGpuBuf *ob = sepia_gpu_alloc(sizeof(float) * (size_t)(T * C), 0);
    if (!ob) die("gpu-forward: sconv: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: sconv: begin failed");
    if (!sepia_gpu_sconv(wb, hb, ib, ob, C, K, T)) die("gpu-forward: sconv: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: sconv: end failed");
    memcpy(out, sepia_gpu_host_ptr(ob), sizeof(float) * (size_t)(T * C));
    sepia_gpu_free(wb);
    sepia_gpu_free(hb);
    sepia_gpu_free(ib);
    sepia_gpu_free(ob);

    if (Km1 > 0) {
        if (T >= Km1) {
            memcpy(hist, in + (size_t)(T - Km1) * (size_t)C, sizeof(float) * (size_t)Km1 * (size_t)C);
        } else {
            int64_t keep = Km1 - T;
            memmove(hist, hist + (size_t)T * (size_t)C, sizeof(float) * (size_t)keep * (size_t)C);
            memcpy(hist + (size_t)keep * (size_t)C, in, sizeof(float) * (size_t)T * (size_t)C);
        }
    }
}

static void gpu_rel_project_call(const float *r_vec, const float *rel_proj, float *rel_logits,
                                  int64_t T, int64_t H, int64_t d_rel, int64_t rel_extent) {
    SepiaGpuBuf *rv = gpu_upload_f32(r_vec, T * H * d_rel);
    SepiaGpuBuf *rp = gpu_upload_f32(rel_proj, d_rel * rel_extent);
    SepiaGpuBuf *rl = sepia_gpu_alloc(sizeof(float) * (size_t)(T * H * rel_extent), 0);
    if (!rl) die("gpu-forward: rel_project: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: rel_project: begin failed");
    if (!sepia_gpu_rel_project(rv, rp, rl, T, H, d_rel, rel_extent)) die("gpu-forward: rel_project: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: rel_project: end failed");
    memcpy(rel_logits, sepia_gpu_host_ptr(rl), sizeof(float) * (size_t)(T * H * rel_extent));
    sepia_gpu_free(rv);
    sepia_gpu_free(rp);
    sepia_gpu_free(rl);
}

static void gpu_banded_attn_call(const float *q, const float *k_full, const float *v_full,
                                  const float *rel_logits, float *attn_out,
                                  const int64_t *kv_lo, const int64_t *kv_hi, const float *tau,
                                  int64_t T, int64_t H, int64_t Hkv, int64_t Dh, int64_t rel_extent,
                                  int64_t q_pos_base, int64_t kv_dim, float inv_head_dim, int64_t cap) {
    SepiaGpuBuf *qb = gpu_upload_f32(q, T * H * Dh);
    SepiaGpuBuf *kb = gpu_upload_f32(k_full, cap * kv_dim);
    SepiaGpuBuf *vb = gpu_upload_f32(v_full, cap * kv_dim);
    SepiaGpuBuf *rl = gpu_upload_f32(rel_logits, T * H * rel_extent);
    SepiaGpuBuf *ob = sepia_gpu_alloc(sizeof(float) * (size_t)(T * H * Dh), 0);
    if (!ob) die("gpu-forward: banded_attn: alloc failed");
    if (!sepia_gpu_begin()) die("gpu-forward: banded_attn: begin failed");
    if (!sepia_gpu_banded_attn(qb, kb, vb, rl, ob, kv_lo, kv_hi, tau, T, H, Hkv, Dh, rel_extent,
                                q_pos_base, kv_dim, inv_head_dim))
        die("gpu-forward: banded_attn: dispatch failed");
    if (!sepia_gpu_end()) die("gpu-forward: banded_attn: end failed");
    memcpy(attn_out, sepia_gpu_host_ptr(ob), sizeof(float) * (size_t)(T * H * Dh));
    sepia_gpu_free(qb);
    sepia_gpu_free(kb);
    sepia_gpu_free(vb);
    sepia_gpu_free(rl);
    sepia_gpu_free(ob);
}

/* GPU-mode twin of attn_forward_chunk (src/sepia.c) -- same nine stages,
 * same cache-mutation semantics (writes lc->k/lc->v/lc->k_hist/lc->v_hist
 * for real, not a snapshot-and-compare like gpu_attn_verify above), just
 * every math step is a sepia_gpu_* dispatch instead of linear/dotf/rmsnorm.
 * kv_lo/kv_hi/tau and the cache-write ordering (write BEFORE the attention
 * loop reads, so self-attention includes the current position) mirror the
 * CPU oracle exactly. */
static void gpu_attn_forward_chunk(const Config *cfg, const LayerWeights *lw, LayerCache *lc,
                                    const float *x_normed, int T, int start_pos, float *out) {
    int hidden = cfg->hidden_size;
    int H = lw->num_heads, Hkv = lw->num_kv_heads, Dh = lw->head_dim;
    int q_dim = lw->q_dim, kv_dim = lw->kv_dim, r_dim = lw->r_dim;
    int K = cfg->conv_kernel_size;
    int d_rel = cfg->d_rel, rel_extent = lw->rel_extent;

    /* 1. projections: wq/wk/wv/wr matvec, one dispatch per (weight, t). */
    float *q_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    float *k_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *r_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)r_dim);
    for (int t = 0; t < T; t++) {
        const float *xt = x_normed + (size_t)t * hidden;
        gpu_matvec_one(lw->wq, xt, q_raw + (size_t)t * q_dim, q_dim, hidden);
        gpu_matvec_one(lw->wk, xt, k_raw + (size_t)t * kv_dim, kv_dim, hidden);
        gpu_matvec_one(lw->wv, xt, v_raw + (size_t)t * kv_dim, kv_dim, hidden);
        gpu_matvec_one(lw->wr, xt, r_raw + (size_t)t * r_dim, r_dim, hidden);
    }

    /* 2. k/v sconv against the layer's persistent history; gpu_sconv_step
     * rolls lc->k_hist/lc->v_hist forward for the NEXT call as a side
     * effect (design choice (b), above). */
    float *k_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    float *v_sconv = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    gpu_sconv_step(lw->k_sconv_w, lc->k_hist, k_raw, k_sconv, kv_dim, K, T);
    gpu_sconv_step(lw->v_sconv_w, lc->v_hist, v_raw, v_sconv, kv_dim, K, T);
    free(k_raw);
    free(v_raw);

    /* 3. per-head RMSNorm: q_raw is [T,H,Dh] contiguous == [T*H,Dh] rows all
     * sharing q_norm[Dh]; k_sconv is [T,Hkv,Dh] == [T*Hkv,Dh] rows sharing
     * k_norm[Dh] -- the same one-dispatch-covers-every-(t,h)-slice trick
     * gpu_attn_verify already relies on. No norm on v (sec.2). */
    float *q_normed = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    float *k_normed = xmalloc(sizeof(float) * (size_t)T * (size_t)kv_dim);
    gpu_rmsnorm_rows(lw->q_norm, q_raw, q_normed, (int64_t)T * H, Dh, (float)cfg->rms_norm_eps);
    gpu_rmsnorm_rows(lw->k_norm, k_sconv, k_normed, (int64_t)T * Hkv, Dh, (float)cfg->rms_norm_eps);
    free(q_raw);
    free(k_sconv);

    /* 4. cache write (normed K, raw-sconv'd V -- no V norm), BEFORE the
     * attention loop below reads lc->k/lc->v, matching attn_forward_chunk's
     * own ordering exactly (self-attention includes the just-written
     * position). */
    for (int t = 0; t < T; t++) {
        memcpy(lc->k + (size_t)(start_pos + t) * kv_dim, k_normed + (size_t)t * kv_dim, sizeof(float) * (size_t)kv_dim);
        memcpy(lc->v + (size_t)(start_pos + t) * kv_dim, v_sconv + (size_t)t * kv_dim, sizeof(float) * (size_t)kv_dim);
    }
    if (start_pos + T > lc->len) lc->len = start_pos + T;
    free(k_normed);
    free(v_sconv);

    /* 5. tau / kv_lo / kv_hi -- host scalars, identical formula to
     * attn_forward_chunk (src/sepia.c). */
    int have_log_scaling = (!lw->is_sliding) && cfg->has_log_scaling_floor;
    int64_t *kv_lo = xmalloc(sizeof(int64_t) * (size_t)T);
    int64_t *kv_hi = xmalloc(sizeof(int64_t) * (size_t)T);
    float *tau = xmalloc(sizeof(float) * (size_t)T);
    for (int t = 0; t < T; t++) {
        int q_pos = start_pos + t;
        double tv = 1.0;
        if (have_log_scaling) {
            double effective_n = (double)(q_pos + 1);
            double ratio = effective_n / (double)cfg->log_scaling_n_floor;
            if (ratio < 1.0) ratio = 1.0;
            tv = 1.0 + cfg->log_scaling_alpha * log(ratio);
        }
        tau[t] = (float)tv;
        int64_t kvlo = lw->is_sliding ? (int64_t)q_pos - (int64_t)cfg->sliding_window_size + 1 : 0;
        if (kvlo < 0) kvlo = 0;
        kv_lo[t] = kvlo;
        kv_hi[t] = q_pos;
    }
    /* tau precision (P2 Task 9 review carry): must match attn_forward_chunk's
     * own formula bit-for-bit -- `(float)((double)q * tau)`, double-precision
     * intermediate, not a plain float*float multiply. This tiny-oracle gate
     * never caught the earlier float-only version because have_log_scaling
     * is never true here (tau==1.0 at every tested position, where the two
     * formulas coincide exactly); Task 9's real model does exercise
     * log-scaling (global layers past the floor position), so the gap is
     * fixed here too, not just in the real-path code that motivated it. */
    float *q_scaled = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < q_dim; i++)
            q_scaled[(size_t)t * q_dim + i] =
                (float)((double)q_normed[(size_t)t * q_dim + i] * (double)tau[t]);
    free(q_normed);

    /* 6. rel-project. */
    float *rel_logits = xmalloc(sizeof(float) * (size_t)T * (size_t)H * (size_t)rel_extent);
    gpu_rel_project_call(r_raw, lw->rel_proj, rel_logits, T, H, d_rel, rel_extent);
    free(r_raw);

    /* 7. banded attention over the full persistent cache [0, start_pos+T) --
     * lc->k/lc->v already hold that whole range after step 4's write. */
    int64_t cap = (int64_t)start_pos + (int64_t)T;
    float *attn_concat = xmalloc(sizeof(float) * (size_t)T * (size_t)q_dim);
    gpu_banded_attn_call(q_scaled, lc->k, lc->v, rel_logits, attn_concat, kv_lo, kv_hi, tau,
                          T, H, Hkv, Dh, rel_extent, start_pos, kv_dim, 1.0f / (float)Dh, cap);
    free(q_scaled);
    free(rel_logits);
    free(kv_lo);
    free(kv_hi);
    free(tau);

    /* 8. wo matvec, one dispatch per t. */
    for (int t = 0; t < T; t++)
        gpu_matvec_one(lw->wo, attn_concat + (size_t)t * q_dim, out + (size_t)t * hidden, hidden, q_dim);
    free(attn_concat);
}

/* GPU-mode twin of mlp_dense_forward. w13 handling (the design choice the
 * task brief called out): dense_w13 is stored interleaved along the OUTPUT-
 * channel axis (row 2i=gate_i, row 2i+1=up_i, sec.9) but is still a single
 * physically contiguous [2*dense_inter,hidden] row-major matrix -- the
 * interleave is a row-index LABELING, not a different memory layout. So the
 * simplest-correct option (of the three the brief offered) is: one
 * sepia_gpu_matvec dispatch over the WHOLE tensor (out_dim=2*dense_inter)
 * produces gu[2*dense_inter] with gu[2i]=gate_i/gu[2i+1]=up_i already
 * sitting at the right offsets (exactly w13_row's own indexing formula,
 * src/sepia.c), then a trivial host gather splits it into g[]/u[] before
 * the silu_mul dispatch. No extra floating-point operation is introduced by
 * the interleave itself -- the gather only moves already-computed floats. */
static void gpu_mlp_dense_forward(const LayerWeights *lw, int hidden, int dense_inter, const float *x, float *out) {
    int two_inter = 2 * dense_inter;
    float *gu = xmalloc(sizeof(float) * (size_t)two_inter);
    gpu_matvec_one(lw->dense_w13, x, gu, two_inter, hidden);

    float *g = xmalloc(sizeof(float) * (size_t)dense_inter);
    float *u = xmalloc(sizeof(float) * (size_t)dense_inter);
    for (int i = 0; i < dense_inter; i++) {
        g[i] = gu[2 * i];
        u[i] = gu[2 * i + 1];
    }
    free(gu);

    float *h = xmalloc(sizeof(float) * (size_t)dense_inter);
    gpu_silu_mul_pair(g, u, h, dense_inter);
    free(g);
    free(u);

    float *y = xmalloc(sizeof(float) * (size_t)hidden);
    gpu_matvec_one(lw->dense_w2, h, y, hidden, dense_inter);
    free(h);

    float gscale = lw->dense_global_scale[0];
    for (int d = 0; d < hidden; d++) out[d] = y[d] * gscale;
    free(y);
}

/* GPU-mode twin of mlp_moe_forward. MoE readback design (the task brief's
 * other called-out choice): the router matvec runs on GPU, then its
 * n_total-length logits are read back and handed to moe_route_select --
 * THE SAME function mlp_moe_forward (CPU path) calls -- for selection and
 * mixing-weight computation. That block is <=10 scalars of sigmoid/log/exp,
 * never a matmul, so reusing the verified CPU routine (rather than
 * reimplementing it against GPU logits) guarantees the GPU path picks
 * IDENTICAL experts with IDENTICAL weights whenever its logits match the
 * CPU's within the dispatch tolerance -- no separate routing logic to drift.
 * Per-expert w13 handling is the same interleaved-matvec-then-gather trick
 * as gpu_mlp_dense_forward, applied to each selected expert's [2*moe_inter,
 * hidden] sub-block (a plain host pointer offset into lw->experts_w13/
 * shared_w13 -- both are already contiguous per-expert blocks, sec.9). */
static void gpu_mlp_moe_forward(const Config *cfg, const LayerWeights *lw, const float *x, float *out) {
    int hidden = cfg->hidden_size;
    int n_routed = cfg->n_routed_experts, n_shared = cfg->n_shared_experts, topk = cfg->num_experts_per_tok;
    int n_total = n_routed + n_shared;
    int moe_inter = cfg->moe_intermediate_size;
    int two_inter = 2 * moe_inter;

    float *router_logits = xmalloc(sizeof(float) * (size_t)n_total);
    gpu_matvec_one(lw->router_w, x, router_logits, n_total, hidden);

    int *topk_idx = xmalloc(sizeof(int) * (size_t)topk);
    int n_sel = topk + n_shared;
    float *weights = xmalloc(sizeof(float) * (size_t)n_sel);
    moe_route_select(cfg, lw, router_logits, topk_idx, weights);
    free(router_logits);

    for (int d = 0; d < hidden; d++) out[d] = 0.0f;

    float *gu = xmalloc(sizeof(float) * (size_t)two_inter);
    float *g = xmalloc(sizeof(float) * (size_t)moe_inter);
    float *u = xmalloc(sizeof(float) * (size_t)moe_inter);
    float *h = xmalloc(sizeof(float) * (size_t)moe_inter);
    float *expert_out = xmalloc(sizeof(float) * (size_t)hidden);

    for (int j = 0; j < topk; j++) {
        int e = topk_idx[j];
        const float *w13_e = lw->experts_w13 + (size_t)e * (size_t)two_inter * (size_t)hidden;
        const float *w2_e = lw->experts_w2 + (size_t)e * (size_t)hidden * (size_t)moe_inter;

        gpu_matvec_one(w13_e, x, gu, two_inter, hidden);
        for (int i = 0; i < moe_inter; i++) {
            g[i] = gu[2 * i];
            u[i] = gu[2 * i + 1];
        }
        gpu_silu_mul_pair(g, u, h, moe_inter);
        gpu_matvec_one(w2_e, h, expert_out, hidden, moe_inter);

        float wj = weights[j];
        for (int d = 0; d < hidden; d++) out[d] += expert_out[d] * wj; /* routed: weight AFTER down_proj (sec.7) */
    }

    for (int s = 0; s < n_shared; s++) {
        const float *w13_s = lw->shared_w13 + (size_t)s * (size_t)two_inter * (size_t)hidden;
        const float *w2_s = lw->shared_w2 + (size_t)s * (size_t)hidden * (size_t)moe_inter;
        float gamma = weights[topk + s];

        gpu_matvec_one(w13_s, x, gu, two_inter, hidden);
        for (int i = 0; i < moe_inter; i++) {
            g[i] = gu[2 * i];
            u[i] = gu[2 * i + 1];
        }
        gpu_silu_mul_pair(g, u, h, moe_inter);
        for (int i = 0; i < moe_inter; i++) h[i] *= gamma; /* shared: gamma multiplies the activation
                                                              * BEFORE down_proj, same order of operations
                                                              * as mlp_moe_forward's own placement (sec.7) */
        gpu_matvec_one(w2_s, h, expert_out, hidden, moe_inter);
        for (int d = 0; d < hidden; d++) out[d] += expert_out[d];
    }

    free(gu);
    free(g);
    free(u);
    free(h);
    free(expert_out);
    free(weights);
    free(topk_idx);
}

/* GPU-mode twin of decoder_layer_forward: same pre-norm residual wiring
 * (attn block -> attn-sconv -> outer residual add; mlp/moe block -> mlp-
 * sconv -> outer residual add), same per-layer sconv history persisted in
 * `lc` across calls. */
static void gpu_decoder_layer_forward(const Config *cfg, const LayerWeights *lw, LayerCache *lc,
                                       float *x /*[T,hidden], in/out*/, int T, int start_pos) {
    int hidden = cfg->hidden_size;
    int K = cfg->conv_kernel_size;

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_rmsnorm_rows(lw->attn_norm, x, normed, T, hidden, (float)cfg->rms_norm_eps);

    float *attn_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_attn_forward_chunk(cfg, lw, lc, normed, T, start_pos, attn_out);
    free(normed);

    float *attn_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_sconv_step(lw->attn_sconv_w, lc->attn_hist, attn_out, attn_sconv_out, hidden, K, T);
    gpu_add_into(x, attn_sconv_out, (int64_t)T * hidden);
    free(attn_out);
    free(attn_sconv_out);

    float *normed2 = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_rmsnorm_rows(lw->mlp_norm, x, normed2, T, hidden, (float)cfg->rms_norm_eps);

    float *mlp_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++) {
        if (lw->is_sparse)
            gpu_mlp_moe_forward(cfg, lw, normed2 + (size_t)t * hidden, mlp_out + (size_t)t * hidden);
        else
            gpu_mlp_dense_forward(lw, hidden, cfg->dense_intermediate_size, normed2 + (size_t)t * hidden,
                                   mlp_out + (size_t)t * hidden);
    }
    free(normed2);

    float *mlp_sconv_out = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_sconv_step(lw->mlp_sconv_w, lc->mlp_hist, mlp_out, mlp_sconv_out, hidden, K, T);
    gpu_add_into(x, mlp_sconv_out, (int64_t)T * hidden);
    free(mlp_out);
    free(mlp_sconv_out);
}

/* GPU-mode twin of model_forward_chunk (forward-declared above that
 * function). Embed lookup is a plain host gather (pointer indexing into the
 * mmap'd embedding table, no math) followed by ONE embed_norm rmsnorm
 * dispatch covering all T rows; final_norm/unembed are the same one-
 * dispatch-per-stage shape; mup-divide and argmax stay on the host per the
 * task brief ("Embed/final-norm/unembed: GPU matvecs; argmax on CPU"). */
static void model_forward_chunk_gpu(const Model *m, Cache *cache, const int *token_ids, int T, int start_pos,
                                     float *logits_out /*[T,unpadded_vocab] or NULL*/) {
    if (!sepia_gpu_available()) die("gpu-forward: requires --metal to have initialized the GPU runtime");
    const Config *cfg = &m->cfg;
    int hidden = cfg->hidden_size;

    float *x_raw = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    for (int t = 0; t < T; t++)
        memcpy(x_raw + (size_t)t * hidden, m->embed + (size_t)token_ids[t] * hidden, sizeof(float) * (size_t)hidden);
    float *x = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_rmsnorm_rows(m->embed_norm, x_raw, x, T, hidden, (float)cfg->rms_norm_eps);
    free(x_raw);

    for (int l = 0; l < cfg->num_hidden_layers; l++)
        gpu_decoder_layer_forward(cfg, &m->layers[l], &cache->layers[l], x, T, start_pos);

    float *normed = xmalloc(sizeof(float) * (size_t)T * (size_t)hidden);
    gpu_rmsnorm_rows(m->final_norm, x, normed, T, hidden, (float)cfg->rms_norm_eps);
    free(x);

    if (logits_out) {
        int unpadded = cfg->unpadded_vocab_size;
        float mup = (float)cfg->logits_mup_width_multiplier;
        float *h = xmalloc(sizeof(float) * (size_t)hidden);
        for (int t = 0; t < T; t++) {
            for (int d = 0; d < hidden; d++) h[d] = normed[(size_t)t * hidden + d] / mup;
            gpu_matvec_one(m->unembed, h, logits_out + (size_t)t * unpadded, unpadded, hidden);
        }
        free(h);
    }
    free(normed);
}

/* --gpu-quants (local-only, needs --metal): the Task 4/5/6 gate for the
 * dequant-fused Q8_0/Q4_K/Q5_K/Q6_K/IQ2_XS/IQ3_XXS/IQ4_XS Metal matvec
 * kernels. For each SQFX fixture (tools/fixtures/quants/, format documented
 * at tools/test_quants.c's run_dequant_fixture): upload the raw quant
 * blocks, run the standalone debug kernel (sepia_gpu_dequant_rows), and
 * require BITWISE equality against the fixture's expected f32 payload --
 * dequantization has no accumulation, so the P2 plan's acceptance policy
 * (Global Constraints (b)) requires the GPU unpack to match the CPU's
 * dequantize_row exactly, not merely within tolerance. Then, for types with
 * a GPU matvec kernel, reinterprets the same fixture's blocks as a small
 * multi-row matrix (a single row would never exercise the row-stride
 * arithmetic, since row 0 always sits at offset 0 regardless of nb1), draws
 * a deterministic random x, and compares sepia_gpu_matvec_q against the CPU
 * qlinear reference at SEPIA_GPU_COMPARE_TOL (the same relative-tolerance
 * gate --gpu-compare-tiny uses). With Task 6 landed, every committed SQFX
 * fixture type now has a GPU kernel; gpu_quants_type_has_kernel stays as a
 * forward-compatible guard (skip-with-notice, not a hard failure) for any
 * future quant type whose fixture is committed before its kernel lands. */

static int g_gpu_quants_failed = 0;

static int gpu_quants_type_has_kernel(int ggml_type) {
    return ggml_type == SEPIA_T_Q8_0 || ggml_type == SEPIA_T_Q4_K ||
           ggml_type == SEPIA_T_Q5_K || ggml_type == SEPIA_T_Q6_K ||
           ggml_type == SEPIA_T_IQ2_XS || ggml_type == SEPIA_T_IQ3_XXS ||
           ggml_type == SEPIA_T_IQ4_XS;
}

static uint32_t gq_rd_u32(FILE *f, const char *path) {
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) die("gpu-quants: short read in %s", path);
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

/* How many rows the fixture's blocks get split into for the matvec check --
 * 4 if n_blocks divides evenly by 4 (both committed fixtures do: q8_0.bin
 * has 64 blocks, q4_k.bin has 16), otherwise falls back to 1 row so any
 * SQFX fixture is still usable, just without row-stride coverage. */
#define SEPIA_GPU_QUANTS_ROWS 4

static void run_gpu_quants_fixture(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die("gpu-quants: cannot open %s", path);
    uint32_t magic = gq_rd_u32(f, path);
    if (magic != 0x58465153u) {
        fprintf(stderr, "gpu-quants: %s: not an SQFX fixture (magic 0x%08x), skipping\n", path, magic);
        fclose(f);
        return;
    }
    uint32_t ver = gq_rd_u32(f, path);
    if (ver != 1) die("gpu-quants: %s: bad SQFX version %u", path, ver);
    uint32_t type = gq_rd_u32(f, path);
    uint32_t n_blocks = gq_rd_u32(f, path);
    uint32_t block_elems = gq_rd_u32(f, path);
    uint32_t block_bytes = gq_rd_u32(f, path);

    if (!gpu_quants_type_has_kernel((int)type)) {
        printf("gpu-quants: %-40s type %2u: no GPU kernel yet, skipping\n", path, type);
        fclose(f);
        return;
    }
    if ((int64_t)block_elems != quants_block_size((int)type) ||
        (size_t)block_bytes != quants_type_size((int)type))
        die("gpu-quants: %s: size table mismatch (type %u: fixture %u/%u, code %lld/%zu)",
            path, type, block_elems, block_bytes,
            (long long)quants_block_size((int)type), quants_type_size((int)type));

    size_t raw_sz = (size_t)n_blocks * block_bytes;
    int64_t n = (int64_t)n_blocks * block_elems;
    uint8_t *raw = xmalloc(raw_sz);
    float *expect = xmalloc((size_t)n * sizeof(float));
    if (fread(raw, 1, raw_sz, f) != raw_sz || fread(expect, sizeof(float), (size_t)n, f) != (size_t)n)
        die("gpu-quants: %s: truncated fixture", path);
    fclose(f);

    /* --- bitwise dequant check (sepia_dequant_rows_<type>) --- */
    SepiaGpuBuf *raw_buf = sepia_gpu_alloc(raw_sz, 0);
    SepiaGpuBuf *dq_buf = sepia_gpu_alloc((size_t)n * sizeof(float), 0);
    if (!raw_buf || !dq_buf) die("gpu-quants: %s: alloc failed", path);
    memcpy(sepia_gpu_host_ptr(raw_buf), raw, raw_sz);

    if (!sepia_gpu_begin()) die("gpu-quants: %s: begin failed (dequant)", path);
    if (!sepia_gpu_dequant_rows((int)type, raw_buf, 0, dq_buf, n))
        die("gpu-quants: %s: dequant_rows dispatch failed", path);
    if (!sepia_gpu_end()) die("gpu-quants: %s: end failed (dequant)", path);

    float *got = (float *)sepia_gpu_host_ptr(dq_buf);
    int fails = 0;
    for (int64_t i = 0; i < n; i++) {
        if (memcmp(&got[i], &expect[i], sizeof(float)) != 0) {
            if (fails < 5) {
                uint32_t gb, eb;
                memcpy(&gb, &got[i], 4);
                memcpy(&eb, &expect[i], 4);
                fprintf(stderr, "  %s elem %lld: got %.9g (0x%08x) expect %.9g (0x%08x)\n",
                        path, (long long)i, (double)got[i], gb, (double)expect[i], eb);
            }
            fails++;
        }
    }
    printf("gpu-quants: %-40s type %2u  %5u blocks  dequant %s\n", path, type, n_blocks, fails ? "FAIL" : "ok");
    if (fails) g_gpu_quants_failed = 1;
    sepia_gpu_free(dq_buf);

    /* --- matvec tolerance check --- */
    uint32_t out_dim = SEPIA_GPU_QUANTS_ROWS;
    while (out_dim > 1 && n_blocks % out_dim != 0) out_dim--;
    int64_t in_dim = (int64_t)(n_blocks / out_dim) * (int64_t)block_elems;

    QTensor w = { (int)type, raw, (int64_t)out_dim, in_dim };
    float *x = xmalloc((size_t)in_dim * sizeof(float));
    unsigned seed = 20260720u ^ type;
    for (int64_t i = 0; i < in_dim; i++)
        x[i] = ((float)rand_r(&seed) / (float)RAND_MAX) * 2.0f - 1.0f;
    float *row_scratch = xmalloc((size_t)in_dim * sizeof(float));
    float *cpu_y = xmalloc((size_t)out_dim * sizeof(float));
    qlinear(&w, x, cpu_y, row_scratch);

    SepiaGpuBuf *x_buf = sepia_gpu_alloc((size_t)in_dim * sizeof(float), 0);
    SepiaGpuBuf *y_buf = sepia_gpu_alloc((size_t)out_dim * sizeof(float), 0);
    if (!x_buf || !y_buf) die("gpu-quants: %s: matvec alloc failed", path);
    memcpy(sepia_gpu_host_ptr(x_buf), x, (size_t)in_dim * sizeof(float));

    if (!sepia_gpu_begin()) die("gpu-quants: %s: matvec begin failed", path);
    if (!sepia_gpu_matvec_q((int)type, raw_buf, 0, x_buf, y_buf, (int64_t)out_dim, in_dim))
        die("gpu-quants: %s: matvec_q dispatch failed", path);
    if (!sepia_gpu_end()) die("gpu-quants: %s: matvec end failed", path);

    float *gpu_y = (float *)sepia_gpu_host_ptr(y_buf);
    double scale = 0.0;
    for (uint32_t j = 0; j < out_dim; j++) scale = fmax(scale, fabs((double)cpu_y[j]));
    double max_rel = 0.0;
    for (uint32_t j = 0; j < out_dim; j++) {
        double rel = gpu_compare_rel_err(gpu_y[j], cpu_y[j], scale);
        if (rel > max_rel) max_rel = rel;
    }
    printf("gpu-quants: %-40s type %2u  matvec %ux%-6lld max_rel_err %.3e  %s\n",
           path, type, out_dim, (long long)in_dim, max_rel,
           max_rel <= SEPIA_GPU_COMPARE_TOL ? "ok" : "FAIL");
    if (max_rel > SEPIA_GPU_COMPARE_TOL) g_gpu_quants_failed = 1;

    sepia_gpu_free(raw_buf);
    sepia_gpu_free(x_buf);
    sepia_gpu_free(y_buf);
    free(raw);
    free(expect);
    free(x);
    free(row_scratch);
    free(cpu_y);
}

static void run_gpu_quants(char **paths, int n_paths) {
    if (!sepia_gpu_available()) die("gpu-quants: requires --metal to have initialized the GPU runtime");
    for (int i = 0; i < n_paths; i++) run_gpu_quants_fixture(paths[i]);
    if (g_gpu_quants_failed) die("gpu-quants: at least one fixture failed");
    printf("gpu-quants ok\n");
}

/* ================================== main =================================== */

int main(int argc, char **argv) {
    const char *dump_path = NULL;
    const char *smoke_dir = NULL;
    const char *prompt = "The capital of France is";
    int n_gen = 32;
    int do_real = 0, do_mlock = 0, logits_only = 0, do_metal = 0, do_gpu_selftest = 0, do_gpu_compare_tiny = 0;
    int do_gpu_quants = 0, do_gpu_compare_attn = 0;
    char **gpu_quants_paths = NULL;
    int gpu_quants_n = 0;
    /* Task 10: default sized well under the ~110GB locked+resident ceiling
     * (Global Constraints) alongside the 14.23GB resident.bin wrap + KV
     * cache/scratch (a few hundred MB) -- see the expert_store_init log
     * line and docs/superpowers/sdd/task-10-report.md for the arithmetic.
     * --expert-cache-gb overrides for local experimentation. */
    double expert_cache_gb = 64.0;
    int verbose_cache = 0;
    int n_repeat = 1;
    /* Task 11 A/B: default stays F_NOCACHE (direct I/O) per Global
     * Constraints -- SEPIA's own measured 13.3GB/s and ~2x-buffered result
     * (docs/ssd-bench.md). --expert-io-mode pagecache switches to ds4's
     * alternative (buffered reads + F_RDADVISE readahead hints) for the A/B. */
    int expert_io_pagecache = 0;

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
        } else if (strcmp(argv[i], "--metal") == 0) {
            do_metal = 1;
        } else if (strcmp(argv[i], "--gpu-selftest") == 0) {
            do_gpu_selftest = 1;
        } else if (strcmp(argv[i], "--gpu-compare-tiny") == 0) {
            do_gpu_compare_tiny = 1;
        } else if (strcmp(argv[i], "--gpu-compare-attn") == 0) {
            do_gpu_compare_attn = 1;
        } else if (strcmp(argv[i], "--expert-cache-gb") == 0 && i + 1 < argc) {
            expert_cache_gb = atof(argv[++i]);
        } else if (strcmp(argv[i], "--verbose-cache") == 0) {
            verbose_cache = 1;
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            n_repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--expert-io-mode") == 0 && i + 1 < argc) {
            const char *mode = argv[++i];
            if (strcmp(mode, "nocache") == 0) expert_io_pagecache = 0;
            else if (strcmp(mode, "pagecache") == 0) expert_io_pagecache = 1;
            else die("--expert-io-mode: unknown mode '%s' (expected nocache or pagecache)", mode);
        } else if (strcmp(argv[i], "--gpu-quants") == 0) {
            do_gpu_quants = 1;
            gpu_quants_paths = &argv[i + 1];
            gpu_quants_n = argc - (i + 1);
            i = argc; /* the rest of argv is fixture paths, not flags */
        } else {
            fprintf(stderr,
                "usage: %s [--dump-acts FILE] [--metal]\n"
                "       %s --smoke DIR\n"
                "       %s --real [--prompt TEXT] [--n-gen N] [--mlock] [--logits-only] [--metal]\n"
                "          [--expert-cache-gb N] [--verbose-cache]\n"
                "       %s --metal --gpu-selftest\n"
                "       %s --metal --gpu-compare-tiny\n"
                "       %s --metal --gpu-compare-attn\n"
                "       %s --metal --gpu-quants FIXTURE.bin [FIXTURE.bin ...]\n"
                "\n"
                "--real paths default to docs/inkling-config.json, weights/resident-manifest.json,\n"
                "weights/inkling-ud-q2_k_xl.sepia-index.json, weights/tokenizer.bin -- override with\n"
                "the SEPIA_REAL_CONFIG_PATH / SEPIA_REAL_MANIFEST_PATH / SEPIA_REAL_INDEX_PATH /\n"
                "SEPIA_REAL_TOKENIZER_PATH env vars.\n"
                "--mlock only makes sense once `extract_resident.py --verify` has confirmed\n"
                "resident.bin's bytes match its manifest -- not enforceable here, the loader trusts its caller.\n"
                "--metal initializes the Metal GPU runtime (metal/*.metal); init failure is fatal since\n"
                "--metal was explicitly requested. For the plain tiny-oracle self-test (no other mode\n"
                "flag), --metal additionally routes the forward pass itself through the GPU kernels\n"
                "(Task 8) instead of the CPU reference path -- --real's forward pass is unaffected (still\n"
                "CPU-only; the real-model GPU path is Task 9).\n"
                "--gpu-selftest exercises the zero-copy GPU buffer API (wrap_mmap/alloc/free/host_ptr)\n"
                "end-to-end against a live device; requires --metal, local-only (needs a Metal GPU).\n"
                "--gpu-compare-tiny runs a real tiny-oracle prefill on the CPU, replays each captured\n"
                "rmsnorm/matvec/silu_mul/add/softmax/sconv instance on the GPU, and reports max relative\n"
                "error per op kind; requires --metal, local-only (needs a Metal GPU).\n"
                "--gpu-compare-attn (Task 7 Gate A) builds synthetic attention geometries at the two\n"
                "real per-layer-type shapes (sliding/global), runs the real attn_forward_chunk as the\n"
                "CPU oracle, and compares the banded-attention Metal kernels against it per (q_pos,n_kv)\n"
                "edge case; requires --metal, local-only (needs a Metal GPU).\n"
                "--gpu-quants runs each SQFX fixture's dequant kernel bitwise against its expected f32\n"
                "payload, plus a tolerance-checked matvec vs CPU qlinear; requires --metal, local-only\n"
                "(needs a Metal GPU).\n"
                "--expert-cache-gb N (--metal --real only) sizes the GPU-resident, per-slot-mlocked,\n"
                "LRU-evicted routed-expert cache (Task 10); default 64 GB. --verbose-cache prints\n"
                "per-step cache hit/miss counts. --repeat N (--real only) repeats generation N times\n"
                "in the SAME process (same RealModel/expert cache) -- run 1 is cold, runs 2+ are warm;\n"
                "also proves double-run determinism when the token sequence matches across runs.\n"
                "--expert-io-mode {nocache|pagecache} (--metal --real only) selects the routed-expert\n"
                "pread path (Task 11 A/B): nocache (default) is F_NOCACHE direct I/O; pagecache is ds4's\n"
                "alternative (buffered reads + an F_RDADVISE readahead hint per region).\n",
                argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
            return 2;
        }
    }

    if (do_metal) {
        if (!sepia_gpu_init("metal")) {
            die("metal: initialization failed (see above)");
        }
        fprintf(stderr, "sepia: metal: initialized (%s)\n", sepia_gpu_device_name());
    }

    if (do_gpu_selftest) {
        run_gpu_selftest();
        return 0;
    }

    if (do_gpu_compare_tiny) {
        run_gpu_compare_tiny();
        return 0;
    }

    if (do_gpu_compare_attn) {
        run_gpu_compare_attn();
        return 0;
    }

    if (do_gpu_quants) {
        if (gpu_quants_n == 0) die("gpu-quants: at least one fixture path required");
        run_gpu_quants(gpu_quants_paths, gpu_quants_n);
        return 0;
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

        size_t expert_cache_budget_bytes = (size_t)(expert_cache_gb * 1e9);

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        RealModel rm = real_load(real_config_path, manifest_path, index_path, tokenizer_path,
                                  expert_cache_budget_bytes, verbose_cache, expert_io_pagecache);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        fprintf(stderr, "real: loaded in %.2fs (resident.bin %.2f GB mmap'd, %d GGUF part(s) open)\n",
                elapsed_ms(t0, t1) / 1000.0, (double)rm.res_size / 1e9, rm.idx.n_parts);

        if (do_mlock) {
            if (mlock(rm.res_base, rm.res_size) != 0) die("mlock: %s", strerror(errno));
            fprintf(stderr, "real: mlock'd resident.bin (%zu bytes)\n", rm.res_size);
        }

        /* --metal routes generation through the real-model GPU resident
         * path (Task 9: quantized weights read straight off gpu_res_buf, no
         * CPU dequant); without --metal, CPU real mode stays byte-for-byte
         * what P1 shipped. do_metal implies rm.gpu_res_buf is set (main()
         * initializes the GPU runtime, dying on failure, before real_load
         * runs). */
        if (logits_only) {
            if (do_metal) real_print_top_logits_gpu(&rm, prompt);
            else real_print_top_logits(&rm, prompt);
            if (do_metal) {
                expert_store_shutdown(&rm.expert_store);
                expert_loader_pool_shutdown();
            }
            return 0;
        }

        if (n_repeat < 1) n_repeat = 1;
        for (int r = 0; r < n_repeat; r++) {
            if (n_repeat > 1)
                fprintf(stderr, "sepia: real: === run %d/%d (%s) ===\n", r + 1, n_repeat, r == 0 ? "cold" : "warm");
            struct timespec r0, r1;
            clock_gettime(CLOCK_MONOTONIC, &r0);
            if (do_metal) real_generate_gpu(&rm, prompt, n_gen);
            else real_generate(&rm, prompt, n_gen);
            clock_gettime(CLOCK_MONOTONIC, &r1);
            if (n_repeat > 1)
                fprintf(stderr, "sepia: real: === run %d/%d wall %.2fs ===\n", r + 1, n_repeat, elapsed_ms(r0, r1) / 1000.0);
        }
        if (do_metal) {
            expert_store_shutdown(&rm.expert_store);
            expert_loader_pool_shutdown();
        }
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

    int ok = run_self_test(&m, ref_path, do_metal);
    return ok ? 0 : 1;
}

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
    return st_;
}

/* Looks up a tensor by its exact on-disk name; dies loudly if missing, of
 * the wrong dtype, or whose byte length disagrees with its declared shape
 * (a cheap but effective shape sanity check without storing full shapes). */
static const float *st_find(const SafeTensors *st, const char *name) {
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
    int unpadded_vocab_size; /* lm_head output width; the padded vocab_size itself is never
                               * needed -- embed/unembed tensor shapes are validated directly
                               * by st_find (sec.8) */

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
    if (lt && lt->type == JSON_ARR && lt->arr_count == (size_t)n) {
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
    if (mlt && mlt->type == JSON_ARR && mlt->arr_count == (size_t)n) {
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
    return st_find(st, name);
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

    m.embed = st_find(&m.st, "model.llm.embed.weight");
    m.embed_norm = st_find(&m.st, "model.llm.embed_norm.weight");
    m.final_norm = st_find(&m.st, "model.llm.norm.weight");
    m.unembed = st_find(&m.st, "model.llm.unembed.weight");

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
    for (int j = 0; j < topk; j++) {
        int e = topk_idx[j];
        for (int i = 0; i < moe_inter; i++) {
            float g = dotf(w13_row(lw->experts_w13, e, two_inter, hidden, 0, i), x, hidden);
            float u = dotf(w13_row(lw->experts_w13, e, two_inter, hidden, 1, i), x, hidden);
            h[i] = silu_f(g) * u;
        }
        float wj = weights[j];
        for (int d = 0; d < hidden; d++) {
            const float *row = lw->experts_w2 + ((size_t)e * hidden + (size_t)d) * moe_inter;
            out[d] += dotf(row, h, moe_inter) * wj; /* routed: weight applied AFTER down_proj (sec.7) */
        }
    }

    for (int s = 0; s < n_shared; s++) {
        float gamma = weights[topk + s];
        for (int i = 0; i < moe_inter; i++) {
            float g = dotf(w13_row(lw->shared_w13, s, two_inter, hidden, 0, i), x, hidden);
            float u = dotf(w13_row(lw->shared_w13, s, two_inter, hidden, 1, i), x, hidden);
            h[i] = silu_f(g) * u * gamma; /* shared: gamma applied to the activation, BEFORE down_proj (sec.7) */
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
static int run_self_test(const Model *m) {
    OracleRef ref = load_oracle_ref(REF_PATH);
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

/* ================================== main =================================== */

int main(int argc, char **argv) {
    const char *dump_path = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump-acts") == 0 && i + 1 < argc) {
            dump_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--dump-acts FILE]\n", argv[0]);
            return 2;
        }
    }

    Model m = model_load(CONFIG_PATH, WEIGHTS_PATH);

    if (dump_path) {
        OracleRef ref = load_oracle_ref(REF_PATH);
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

    int ok = run_self_test(&m);
    return ok ? 0 : 1;
}

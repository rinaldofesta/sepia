/* Fixture-gated tokenizer test.
 * Usage: ./test_tokenizer <sidecar.bin> <cases.json>
 * cases.json: [{"text": "...", "ids": [..]}, ...]  (tools/make_tokenizer_fixtures.py)
 * Gate: exact id-sequence equality on encode; exact byte equality on decode. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include "../src/tokenizer.h"

/* Minimal JSON reading for the fixed cases shape, no dependency on sepia.c:
 * find "text" strings and "ids" arrays with a hand scanner. The fixture
 * generator writes JSON with indent=1, ensure_ascii=False: each case is
 * {"text": <string>, "ids": [<int>, ...]}. Strings may contain \", \\, \n,
 * \t, \r, \uXXXX (incl. UTF-16 surrogate pairs), and raw multibyte UTF-8
 * bytes (only control chars and " \ are ever backslash-escaped by the
 * generator). We scan for the literal keys "text" and "ids" in sequence,
 * JSON-unescape the text value to UTF-8 bytes, and strtol the ids array. */

#define MAX_IDS 8192
#define DECODE_CAP 65536

static void perr(const char *fmt, ...) {
    va_list ap; va_start(ap, fmt);
    fprintf(stderr, "test_tokenizer: "); vfprintf(stderr, fmt, ap); fprintf(stderr, "\n");
    va_end(ap); exit(2);
}

static char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) perr("cannot open %s", path);
    if (fseek(f, 0, SEEK_END) != 0) perr("seek failed on %s", path);
    long sz = ftell(f);
    if (sz < 0) perr("ftell failed on %s", path);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) perr("oom reading %s", path);
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) perr("short read on %s", path);
    fclose(f);
    buf[sz] = 0;
    if (out_len) *out_len = (size_t)sz;
    return buf;
}

static const char *skip_ws(const char *p) {
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r') p++;
    return p;
}

/* Parse exactly 4 hex digits at *pp, advance *pp past them. */
static uint32_t parse_hex4(const char **pp) {
    const char *p = *pp;
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) {
        char c = p[i];
        int d = 0;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else perr("bad \\u escape (expected 4 hex digits)");
        v = v * 16 + (uint32_t)d;
    }
    *pp = p + 4;
    return v;
}

/* Encode a Unicode code point as UTF-8 into *wp, advancing it. */
static void append_utf8(uint32_t cp, uint8_t **wp) {
    uint8_t *w = *wp;
    if (cp < 0x80) {
        *w++ = (uint8_t)cp;
    } else if (cp < 0x800) {
        *w++ = (uint8_t)(0xC0 | (cp >> 6));
        *w++ = (uint8_t)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        *w++ = (uint8_t)(0xE0 | (cp >> 12));
        *w++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (uint8_t)(0x80 | (cp & 0x3F));
    } else {
        *w++ = (uint8_t)(0xF0 | (cp >> 18));
        *w++ = (uint8_t)(0x80 | ((cp >> 12) & 0x3F));
        *w++ = (uint8_t)(0x80 | ((cp >> 6) & 0x3F));
        *w++ = (uint8_t)(0x80 | (cp & 0x3F));
    }
    *wp = w;
}

/* *pp must point at the opening '"'. Unescapes into out (caller-sized,
 * at least strlen(*pp)+1 bytes is always enough since escapes never grow
 * the byte count). Advances *pp past the closing '"'. Returns byte count
 * written (not NUL-terminated by this function). */
static size_t unescape_string(const char **pp, uint8_t *out) {
    const char *p = *pp;
    if (*p != '"') perr("expected opening '\"' for string value");
    p++;
    uint8_t *w = out;
    for (;;) {
        unsigned char c = (unsigned char)*p;
        if (c == 0) perr("unterminated string (missing closing '\"')");
        if (c == '"') { p++; break; }
        if (c == '\\') {
            p++;
            char e = *p;
            switch (e) {
            case '"':  *w++ = '"';  p++; break;
            case '\\': *w++ = '\\'; p++; break;
            case '/':  *w++ = '/';  p++; break;
            case 'n':  *w++ = '\n'; p++; break;
            case 't':  *w++ = '\t'; p++; break;
            case 'r':  *w++ = '\r'; p++; break;
            case 'b':  *w++ = '\b'; p++; break;
            case 'f':  *w++ = '\f'; p++; break;
            case 'u': {
                p++;
                uint32_t cp = parse_hex4(&p);
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    if (p[0] != '\\' || p[1] != 'u') perr("lone UTF-16 high surrogate");
                    p += 2;
                    uint32_t lo = parse_hex4(&p);
                    if (lo < 0xDC00 || lo > 0xDFFF) perr("invalid UTF-16 low surrogate");
                    cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                    perr("lone UTF-16 low surrogate");
                }
                append_utf8(cp, &w);
                break;
            }
            default:
                perr("bad escape sequence '\\%c' in string", e);
            }
        } else {
            *w++ = (uint8_t)c;
            p++;
        }
    }
    *pp = p;
    return (size_t)(w - out);
}

static void print_ids(const int32_t *ids, int n) {
    for (int i = 0; i < n; i++) printf("%s%d", i ? ", " : "", ids[i]);
}

static void run_case(const Tokenizer *t, int idx,
                     const uint8_t *text, size_t text_len,
                     const int32_t *want_ids, int n_want,
                     const char *label, int label_len, int *n_ok) {
    int32_t got_ids[MAX_IDS];
    int n_got = tokenizer_encode(t, (const char *)text, got_ids, MAX_IDS);
    int enc_ok = (n_got == n_want);
    if (enc_ok) {
        for (int i = 0; i < n_got; i++) {
            if (got_ids[i] != want_ids[i]) { enc_ok = 0; break; }
        }
    }

    char decode_buf[DECODE_CAP];
    tokenizer_decode(t, want_ids, n_want, decode_buf, sizeof decode_buf);
    int dec_ok = strlen(decode_buf) == text_len && memcmp(decode_buf, text, text_len) == 0;

    if (enc_ok && dec_ok) {
        printf("case %3d ok    %.*s\n", idx, label_len, label);
        (*n_ok)++;
        return;
    }
    printf("case %3d FAIL  %.*s\n", idx, label_len, label);
    if (!enc_ok) {
        printf("  encode: got [");
        print_ids(got_ids, n_got);
        printf("] want [");
        print_ids(want_ids, n_want);
        printf("]\n");
    }
    if (!dec_ok) {
        printf("  decode: got %zu bytes, want %zu bytes (text mismatch)\n",
               strlen(decode_buf), text_len);
    }
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: test_tokenizer <sidecar.bin> <cases.json>\n");
        return 2;
    }
    const char *sidecar_path = argv[1];
    const char *cases_path = argv[2];

    Tokenizer *t = tokenizer_load(sidecar_path); /* dies on format error */

    char *buf = read_file(cases_path, NULL);
    const char *p = skip_ws(buf);
    if (*p != '[') perr("%s: expected top-level JSON array", cases_path);

    int n_total = 0, n_ok = 0;
    const char *scan = p;
    for (;;) {
        const char *tk = strstr(scan, "\"text\"");
        if (!tk) break;

        const char *vp = strchr(tk, ':');
        if (!vp) perr("%s: missing ':' after \"text\"", cases_path);
        vp++; vp = skip_ws(vp);
        if (*vp != '"') perr("%s: \"text\" value is not a string", cases_path);

        const char *text_json_start = vp;
        uint8_t *text_buf = malloc(strlen(vp) + 1);
        if (!text_buf) perr("oom");
        const char *after_text = vp;
        size_t text_len = unescape_string(&after_text, text_buf);
        text_buf[text_len] = 0;
        const char *text_json_end = after_text;

        const char *ik = strstr(after_text, "\"ids\"");
        if (!ik) perr("%s: missing \"ids\" after \"text\"", cases_path);
        const char *ivp = strchr(ik, ':');
        if (!ivp) perr("%s: missing ':' after \"ids\"", cases_path);
        ivp++; ivp = skip_ws(ivp);
        if (*ivp != '[') perr("%s: \"ids\" value is not an array", cases_path);
        ivp++;

        int32_t want_ids[MAX_IDS];
        int n_want = 0;
        for (;;) {
            ivp = skip_ws(ivp);
            if (*ivp == ',') { ivp++; continue; }
            if (*ivp == ']') { ivp++; break; }
            if (*ivp == 0) perr("%s: unterminated \"ids\" array", cases_path);
            char *endp;
            long v = strtol(ivp, &endp, 10);
            if (endp == ivp) perr("%s: bad integer in \"ids\" array", cases_path);
            if (n_want >= MAX_IDS) perr("%s: \"ids\" array exceeds test cap %d", cases_path, MAX_IDS);
            want_ids[n_want++] = (int32_t)v;
            ivp = endp;
        }
        scan = ivp;

        n_total++;
        run_case(t, n_total, text_buf, text_len, want_ids, n_want,
                 text_json_start, (int)(text_json_end - text_json_start), &n_ok);
        free(text_buf);
    }
    if (n_total == 0) perr("%s: no cases found", cases_path);

    tokenizer_free(t);
    free(buf);

    printf("test_tokenizer: %d/%d cases ok (%s)\n", n_ok, n_total, cases_path);
    return n_ok == n_total ? 0 : 1;
}

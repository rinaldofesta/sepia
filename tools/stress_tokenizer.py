#!/usr/bin/env python3
"""Reproducible tokenizer stress sweep (P2 close-out; P1 deferral, see
.superpowers/sdd/progress.md's P1 final-review deferred-items note).

Dev-only: .venv/bin/python tools/stress_tokenizer.py (needs the `tokenizers`
package, exactly like tools/make_tokenizer_fixtures.py -- NOT stdlib). Needs
weights/tokenizer.bin (Task 7's SPTK sidecar) and a built ./test_tokenizer
(built automatically below via `make test_tokenizer` if missing).

What it does: generates a large, deterministic corpus of random strings that
mix the pretokenizer scanner's branch classes (src/tokenizer.c's
next_pretoken: letter runs incl. mixed-script/combining-mark edge cases,
digit runs, punctuation runs, whitespace runs incl. NBSP/line-separator,
contractions), encodes each string with the HF reference tokenizer built
from the SAME weights/tokenizer.bin sidecar the C engine reads (reusing
tools/make_tokenizer_fixtures.py's read_sptk/build_reference verbatim, not
reimplementing byte-level BPE), writes them out in the same {"text","ids"}
shape as tools/fixtures/tokenizer/real_cases.json, and hands that file to
the committed ./test_tokenizer C driver -- which independently re-encodes
every case with the real C tokenizer and asserts id-sequence equality
(plus its own decode-roundtrip check). This is a corpus generator, not a
new comparison engine: the actual C-vs-reference comparison is exactly the
one test_tokenizer.c already performs for the small committed fixtures.

Note on provenance: the original P1 stress-sweep generator that produced
the "~8300-string stress sweep, zero mismatches" figure cited in Task 10's
report lived only in a now-overwritten gitignored scratch report and was
never committed (P1 final review recorded this as deferred work, not lost
data). This script re-derives the corpus from the scanner's branch classes
per that deferral's own fallback instruction; it is a fresh, independently
written generator, not a recovery of the original.

Determinism: SEED below is fixed forever. Do not change it -- a stable seed
is what makes "the stress sweep" a reproducible artifact instead of a fresh
fuzz run every time. If the scanner or reference tokenizer ever changes in
a way that flips a result, that's a real regression signal specifically
because the corpus itself did not change.
"""
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
from make_tokenizer_fixtures import read_sptk, build_reference  # noqa: E402

SIDECAR = REPO / "weights" / "tokenizer.bin"
TEST_TOKENIZER = REPO / "test_tokenizer"

SEED = 20260721          # fixed forever; see determinism note above
N_STRINGS = 8300         # matches the figure cited in README/progress.md

# ---- Branch-class fragment vocab -----------------------------------------
# Codepoint pools below are chosen to exercise src/tokenizer.c's next_pretoken
# branches (comments there use the same branch numbering):
#   branch 1+2: letter run  A*B+ | A+B*  (A = (L|M) minus ASCII a-z,
#               B = (L|M) minus ASCII A-Z; every non-ASCII letter/mark is in
#               BOTH classes -- the interesting edge cases are ASCII
#               case-boundary transitions and combining marks) + optional
#               trailing contraction.
#   branch 3:   \p{N}{1,3} (digit runs get chunked 3 at a time).
#   branch 4:   ' '? [^\s L N]+ [\r\n/]* (punctuation/symbol runs).
#   branch 5/6/7: whitespace runs, incl. the \s*[\r\n]+ vs \s+(?!\S) vs \s+
#               distinction (trailing non-newline ws after the last
#               newline is NOT consumed by branch 5).

ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
ASCII_UPPER = ASCII_LOWER.upper()

# Combining Diacritical Marks block (a representative subset): acute, grave,
# tilde, diaeresis, cedilla, ring above, circumflex.
COMBINING_MARKS = [0x0301, 0x0300, 0x0303, 0x0308, 0x0327, 0x030A, 0x0302]

# Non-Latin letter blocks: (lo, hi) ranges of real letter codepoints (careful
# to avoid punctuation-shaped codepoints embedded in some blocks).
LETTER_BLOCKS = [
    (0x00C0, 0x00FF),   # Latin-1 Supplement letters (café/naïve/etc range)
    (0x0100, 0x017F),   # Latin Extended-A
    (0x0391, 0x03A9),   # Greek uppercase
    (0x03B1, 0x03C9),   # Greek lowercase
    (0x0410, 0x044F),   # Cyrillic
    (0x05D0, 0x05EA),   # Hebrew
    (0x0621, 0x064A),   # Arabic letters
    (0x0904, 0x0939),   # Devanagari
    (0x3041, 0x3096),   # Hiragana
    (0x30A1, 0x30FA),   # Katakana
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0xAC00, 0xD7A3),   # Hangul syllables
]

# Digit pools: ASCII, and non-ASCII \p{N} codepoints (Nd/Nl/No all count,
# see tools/make_unicode_tables.py's classify()).
ASCII_DIGITS = "0123456789"
UNICODE_ND_BLOCKS = [
    (0x0660, 0x0669),   # Arabic-Indic digits
    (0x0966, 0x096F),   # Devanagari digits
    (0xFF10, 0xFF19),   # Fullwidth digits
]
UNICODE_NUMBER_SINGLETONS = [0x2160, 0x2170, 0x00B2, 0x00B3, 0x2155]  # Nl/No

ASCII_PUNCT = list("!?.,;:'\"-_/\\()[]{}@#$%^&*+=<>|~`")
UNICODE_SYMBOLS = [
    "—", "–", "“", "”", "…",   # — – “ ” …
    "©", "®", "™", "§", "¶",   # © ® ™ § ¶
    "€", "£", "¥", "½", "№",   # € £ ¥ ½ №
    "←", "→", "↑", "↓",             # ← → ↑ ↓
    "\U0001F642", "\U0001F680", "\U0001F44D\U0001F3FD",  # 🙂 🚀 👍🏽
    "\U0001F468‍\U0001F469‍\U0001F467",       # ZWJ family emoji
]

# Whitespace codepoints: exactly tools/make_unicode_tables.py's WHITESPACE
# list (Unicode White_Space property, not str.isspace()), so this sweep
# genuinely covers what the C table encodes.
WHITESPACE_CPS = (
    [0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20, 0x85, 0xA0, 0x1680]
    + list(range(0x2000, 0x200B))
    + [0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
)

CONTRACTION_SUFFIXES = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
NON_CONTRACTIONS = ["'x", "'q", "'z9", "'", "'k"]  # should NOT match branch 1/2's tail


def _rand_case(rng, s):
    return "".join(c.upper() if rng.random() < 0.5 else c for c in s)


def frag_letters(rng):
    """One letter run: pure ASCII (upper/lower/mixed-case), a non-Latin
    script word, or a base letter with combining marks attached (incl. the
    'bare combining mark' edge case the scanner comment calls out)."""
    kind = rng.randrange(5)
    if kind == 0:
        return "".join(rng.choice(ASCII_LOWER) for _ in range(rng.randint(1, 10)))
    if kind == 1:
        return "".join(rng.choice(ASCII_UPPER) for _ in range(rng.randint(1, 10)))
    if kind == 2:
        # mixed ASCII case -- maximizes A/B class alternation (camelCase-like)
        return _rand_case(rng, "".join(rng.choice(ASCII_LOWER) for _ in range(rng.randint(2, 10))))
    if kind == 3:
        lo, hi = rng.choice(LETTER_BLOCKS)
        return "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 8)))
    # base letter(s) + combining mark(s), sometimes a bare combining mark
    if rng.random() < 0.15:
        return chr(rng.choice(COMBINING_MARKS))  # bare mark, no base letter
    out = []
    for _ in range(rng.randint(1, 5)):
        out.append(rng.choice(ASCII_LOWER + ASCII_UPPER))
        if rng.random() < 0.6:
            out.append(chr(rng.choice(COMBINING_MARKS)))
    return "".join(out)


def frag_digits(rng):
    kind = rng.randrange(3)
    if kind == 0:
        return "".join(rng.choice(ASCII_DIGITS) for _ in range(rng.randint(1, 10)))
    if kind == 1:
        lo, hi = rng.choice(UNICODE_ND_BLOCKS)
        return "".join(chr(rng.randint(lo, hi)) for _ in range(rng.randint(1, 6)))
    return chr(rng.choice(UNICODE_NUMBER_SINGLETONS))


def frag_punct(rng):
    lead = " " if rng.random() < 0.3 else ""
    if rng.random() < 0.6:
        body = "".join(rng.choice(ASCII_PUNCT) for _ in range(rng.randint(1, 6)))
    else:
        body = "".join(rng.choice(UNICODE_SYMBOLS) for _ in range(rng.randint(1, 3)))
    trail = "".join(rng.choice("\r\n/") for _ in range(rng.randint(0, 3)))
    return lead + body + trail


def frag_whitespace(rng):
    return "".join(chr(rng.choice(WHITESPACE_CPS)) for _ in range(rng.randint(1, 6)))


def frag_contraction(rng):
    """A letter run immediately followed by a contraction suffix (or a
    look-alike that must NOT be treated as one)."""
    base = frag_letters(rng)
    if not base or not base[-1].isascii() or not base[-1].isalpha():
        base = "".join(rng.choice(ASCII_LOWER) for _ in range(rng.randint(2, 6)))
    suffix = rng.choice(CONTRACTION_SUFFIXES if rng.random() < 0.85 else NON_CONTRACTIONS)
    if suffix in CONTRACTION_SUFFIXES:
        suffix = _rand_case(rng, suffix)
    return base + suffix


FRAGMENT_GENS = [frag_letters, frag_digits, frag_punct, frag_whitespace, frag_contraction]


def gen_string(rng):
    n_frags = rng.randint(1, 6)
    parts = []
    for _ in range(n_frags):
        gen = rng.choice(FRAGMENT_GENS)
        parts.append(gen(rng))
        if rng.random() < 0.25:
            parts.append(" ")  # occasional realistic word-boundary space
    return "".join(parts)


def generate_corpus(seed, n):
    rng = random.Random(seed)
    seen_valid = []
    while len(seen_valid) < n:
        s = gen_string(rng)
        s.encode("utf-8")  # guards against any stray surrogate codepoint
        seen_valid.append(s)
    return seen_valid


def main():
    if not SIDECAR.exists():
        print(f"stress_tokenizer: missing {SIDECAR} (needs the real weights "
              "sidecar; this is a local-only tool, same prerequisite as "
              "`make tokreal`)", file=sys.stderr)
        return 1

    subprocess.run(["make", "test_tokenizer"], cwd=REPO, check=True)

    print(f"stress_tokenizer: building HF reference tokenizer from {SIDECAR}")
    toks_raw, ttypes, merges, _bos, _eos = read_sptk(SIDECAR)
    ref = build_reference(toks_raw, ttypes, merges)

    print(f"stress_tokenizer: generating {N_STRINGS} strings (seed={SEED})")
    corpus = generate_corpus(SEED, N_STRINGS)

    cases = []
    roundtrip_failures = []
    for text in corpus:
        ids = ref.encode(text).ids
        decoded = ref.decode(ids, skip_special_tokens=False)
        if decoded != text and text != "":
            roundtrip_failures.append(text)
            continue
        cases.append({"text": text, "ids": ids})

    if roundtrip_failures:
        print(f"stress_tokenizer: STOP -- {len(roundtrip_failures)} strings "
              "failed to roundtrip through the HF REFERENCE tokenizer itself "
              "(before the C tokenizer is even involved) -- this is a corpus "
              "or reference-tokenizer bug, not a C-tokenizer finding. First "
              f"failure: {roundtrip_failures[0]!r}", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        cases_path = Path(td) / "stress_cases.json"
        cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=1) + "\n")
        print(f"stress_tokenizer: wrote {len(cases)} cases to {cases_path}")
        print("stress_tokenizer: running ./test_tokenizer ...")
        result = subprocess.run(
            [str(TEST_TOKENIZER), str(SIDECAR), str(cases_path)],
            cwd=REPO, capture_output=True, text=True,
        )

    # test_tokenizer prints one "case N ok/FAIL ..." line per case plus a
    # final "test_tokenizer: N/M cases ok (...)" summary; forward FAIL lines
    # (if any) and the summary, not every "ok" line (8300 of those is noise).
    fail_lines = [ln for ln in result.stdout.splitlines() if " FAIL " in ln]
    summary = next((ln for ln in result.stdout.splitlines()
                    if ln.startswith("test_tokenizer:")), None)
    for ln in fail_lines:
        print(ln)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    print(summary or "stress_tokenizer: no summary line from test_tokenizer (unexpected)")

    if result.returncode != 0:
        print("stress_tokenizer: MISMATCH -- the C tokenizer diverged from "
              "the HF reference on at least one generated string. This is a "
              "real finding: do not narrow the corpus to make it pass; "
              "investigate the failing case(s) above.", file=sys.stderr)
        return 1

    print(f"stress_tokenizer: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

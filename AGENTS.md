# Engineering rules for SEPIA

These rules bind humans and AI agents working on this repo equally.

## Stack

- The engine is C11 plus Metal (MSL) kernels with an Objective-C runtime
  shim. No C++ in the engine, ever. Python is for offline tooling only
  (converter, oracle, eval); nothing Python runs in the inference hot path.
- Plain `make`. No CMake, no package managers in the engine build.

## Correctness discipline

- The oracle is the gate: no change to the forward pass merges unless
  `./sepia` (no args) reports the self-test token-exact. If you change
  what the model computes, you regenerate the oracle and say why.
- Quantization math in Python tools must be bit-identical to the C engine
  (`np.rint` == `lrintf`, same packing). The converter proves it with
  byte-compares, not by inspection.
- Performance claims come with the exact command, the machine, and the
  numbers. No adjectives where a measurement fits.

## Code quality

From ds4's AGENT.md, adopted verbatim as house rule: "Don't settle for the
first thing that comes to mind, try to find the most minimal and better
working design. Don't introduce slop: very fragile code that just patches
specific cases, dead code, useless code and code ways more complicated of
how it should be."

- Port battle-tested code from ds4/colibri with attribution (NOTICE plus a
  source-file comment) instead of rewriting it worse.
- One model family, done well. Genericity is not a goal; a second
  architecture gets its own file, not an abstraction layer.

## Repository hygiene

- Weights, containers, imatrix data: never in git. The tiny oracle model
  (a few MB, needed by CI, lives under `tools/oracle/`) is the one
  exception, and `.gitignore`'s negation rule names that exact path.
- Measured results go in `docs/` as dated markdown with the commands to
  reproduce them.

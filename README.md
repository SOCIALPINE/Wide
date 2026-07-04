# wide

> **Illuminate, don't isolate.** — a language that *shows* costs and risks instead of forbidding or hiding them.

wide is an experimental programming language (implemented in Rust) built around one idea: the dangers
and costs that other languages wall off (`unsafe`), promise away ("zero-cost"), or silently hide
(GPU transfers) should instead be **illuminated** — tracked by the compiler and shown next to your code.

```
$ cargo run -- examples/borrow.wide

  r = &xs
      INFO: shared borrow of xs statically proven safe — cost 0 (no runtime guard)
  xs.push(2)
      WARN: borrow conflict: push on xs needs &mut, but xs is shared-borrowed — &mut XOR &
```

## What works today (v0.51)

- **A full general-purpose core** — functions, closures/first-class functions, structs/enums/pattern
  matching, error values with `?`, modules, file I/O, slices, string tools. Two backends: a
  tree-walking interpreter (reference) and a bytecode VM (`--vm`) with verified parity.
- **The borrow gradient, all three tiers real** — `&x` / `&mut x` with scope lifetimes:
  *proof* (compile-time safety proof → no runtime guard, definite conflicts caught before run,
  zero false positives) → *guard* (runtime check) → *trust* (`@trust`, illuminated). Plus pointers
  with provenance, `raw.*` access that warns instead of blocking, and `@show provenance`.
- **A JIT that makes "dynamic + native speed" real** — numeric functions (i64/f64, including
  recursion) compile to actual machine code via Cranelift (`--features jit`): fib(30) in **7 ms** —
  19× faster than Python, within 3–7× of `rustc -O` / `gcc -O2`, ~900× its own interpreter
  ([BENCH.md](BENCH.md) has the full table and the honest caveats).
- **A see-through PyTorch** — tensors, broadcasting, autodiff (backprop tape), SGD/Adam, conv2d /
  maxpool2d / softmax — a CNN trains end to end. Static *shape checking* catches matmul mismatches
  and unifies symbolic dimensions (`fn layer(x: tensor[(B,K)], w: tensor[(K,N)])`) at compile time.
- **A real GPU backend** (`--features gpu`, wgpu) — matmul and elementwise ops run as actual WGSL
  compute shaders; chained ops stay resident on the device (zero re-uploads, illuminated). Honest
  numbers: the GPU wins at 1024² (2.9×) and *loses* at 512² — and the illumination shows you why.

Every cost — heap allocations, FLOPs, transfers, activation memory, borrows, file I/O bytes — flows
through one illumination channel (`INFO:`/`WARN:`) woven into your source lines after each run.

## Install

**Prebuilt binaries** (Windows / Linux / macOS) are attached to each
[GitHub Release](../../releases) — download, unpack, and run:

```bash
wide examples/hello.wide
wide --help
```

The binary is standalone (the JIT and GPU backends are compiled in and degrade gracefully —
no GPU means an illuminated CPU fallback, not a failure).

**From source** (needs Rust):

```bash
git clone <this repo> && cd wide
cargo run -- examples/hello.wide
```

The default build has zero external dependencies and compiles in well under a minute.

## Quick start

```bash
cargo run -- examples/hello.wide          # run a program (static check → run → illumination report)
cargo run -- --vm examples/control.wide   # bytecode VM backend
cargo run --features jit -- examples/jit.wide     # native JIT (integer/float functions → machine code)
cargo run --features gpu -- examples/ai/gpu.wide  # real GPU compute (wgpu)
cargo test                                # 166 tests (default); slim/jit/gpu suites also available
python bench/run.py                       # reproduce the benchmarks
```

Start with the guide: **[GUIDE.en.md](GUIDE.en.md)** (English) / **[GUIDE.ko.md](GUIDE.ko.md)** (한국어).
28 runnable examples live in [examples/](examples/).

## Documents

| File | What it is |
|------|------------|
| [GUIDE.en.md](GUIDE.en.md) | The programming guide — start here |
| [GUIDE.ko.md](GUIDE.ko.md) | 프로그래밍 가이드 (한국어) |
| [BENCH.md](BENCH.md) | Measured performance — methodology and honest caveats included (한국어) |
| [examples/](examples/) | 28 runnable, commented example programs |

## Honesty ledger

This project does not claim what it cannot show. Current known limits: GPU results are read back
eagerly after each op (lazy downloads pending) and broadcasts fall back to the CPU; the JIT covers
numeric functions only; closures, explicit borrows, and tensors run on the tree-walker only (VM
parity pending); the static borrow proof is deliberately conservative (unprovable borrows demote
to a runtime guard rather than being rejected); `.backward()` requires a scalar loss and division
is not differentiable. When in doubt, run it — the illumination tells you what actually happened.

## License

Not yet chosen — this is a design-study project. Open an issue if you want to build on it.

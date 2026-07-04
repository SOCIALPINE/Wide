#!/usr/bin/env python3
"""wide benchmark harness — honest numbers for BENCH.md.

Builds two release binaries (default = tree-walker+VM; --features jit = tree-walker with the
Cranelift JIT active) and times each benchmark on each backend, best of N runs. Also times the
equivalent Python program when python3 is available, as an external reference point.

Usage:  python bench/run.py            (from the repo root)
"""
import subprocess, sys, time, os, statistics

RUNS = 3
BENCHES = [
    ("fib.wide", "fib(30) recursion"),
    ("loop.wide", "10M-iteration while loop"),
    ("floatloop.wide", "10M-iteration f64 while loop"),
    ("matmul.wide", "matmul 128x128 x20 (bulk op)"),
]
PY_EQUIV = {
    "fib.wide": "import sys;sys.setrecursionlimit(10000)\ndef fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\nprint(fib(30))",
    "loop.wide": "total = 0\ni = 0\nwhile i < 10000000:\n    total += i\n    i += 1\nprint(total)",
    "floatloop.wide": "total = 0.0\ni = 0.0\nwhile i < 10000000.0:\n    total += i\n    i += 1.0\nprint(total)",
}
# Equivalent C and Rust programs (bench/native/*.{c,rs}), self-timed the same way ("time: N ms" on
# stderr). Compiled with -O2 / -O when gcc/rustc are on PATH; columns show n/a otherwise.
NATIVE = {
    "fib.wide": "fib",
    "loop.wide": "intloop",
    "floatloop.wide": "floatloop",
}

def build():
    print("building release binaries ...")
    subprocess.run(["cargo", "build", "--release"], check=True, capture_output=True)
    exe_plain = os.path.join("target", "release", "wide.exe")
    tmp = exe_plain + ".novjit"
    os.replace(exe_plain, tmp)
    subprocess.run(["cargo", "build", "--release", "--features", "jit"], check=True, capture_output=True)
    exe_jit = exe_plain + ".jit"
    os.replace(exe_plain, exe_jit)
    os.replace(tmp, exe_plain)
    return exe_plain, exe_jit

def timed_wide(cmd):
    """In-process time via `--time` (stderr `time: N ms`) — excludes process startup, which on this
    machine is dominated by antivirus exe scanning (~350ms wall, ~0 CPU) and would poison the numbers."""
    best = None
    for _ in range(RUNS):
        r = subprocess.run(cmd + ["--time"], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        for ln in r.stderr.splitlines():
            if ln.startswith("time:"):
                dt = float(ln.split()[1]) / 1000.0
                best = dt if best is None else min(best, dt)
    return best

def which(exe):
    from shutil import which as _w
    return _w(exe)

def build_native():
    """Compile the C (gcc -O2) and Rust (rustc -O) references into bench/native/build/."""
    out = {}
    bdir = os.path.join("bench", "native", "build")
    os.makedirs(bdir, exist_ok=True)
    gcc = which("gcc")
    rustc = which("rustc")
    ext = ".exe" if os.name == "nt" else ""
    for name in set(NATIVE.values()):
        entry = {}
        if gcc:
            exe = os.path.join(bdir, f"{name}_c{ext}")
            r = subprocess.run([gcc, "-O2", os.path.join("bench", "native", f"{name}.c"), "-o", exe], capture_output=True)
            if r.returncode == 0:
                entry["c"] = exe
        if rustc:
            exe = os.path.join(bdir, f"{name}_rs{ext}")
            r = subprocess.run([rustc, "-O", os.path.join("bench", "native", f"{name}.rs"), "-o", exe], capture_output=True)
            if r.returncode == 0:
                entry["rust"] = exe
        out[name] = entry
    return out

def timed_native(exe):
    """A self-timed native reference — parse its `time: N ms` line."""
    if exe is None:
        return None
    best = None
    for _ in range(RUNS):
        r = subprocess.run([exe], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        for ln in r.stderr.splitlines():
            if ln.startswith("time:"):
                dt = float(ln.split()[1]) / 1000.0
                best = dt if best is None else min(best, dt)
    return best

def timed_py(code):
    """Python comparison, timed in-process the same way (time.perf_counter around the work)."""
    wrapped = "import time as _t\n_t0=_t.perf_counter()\n" + code + "\nimport sys;print('time:',(_t.perf_counter()-_t0)*1000,file=sys.stderr)"
    best = None
    for _ in range(RUNS):
        r = subprocess.run([sys.executable, "-c", wrapped], capture_output=True, text=True)
        if r.returncode != 0:
            return None
        for ln in r.stderr.splitlines():
            if ln.startswith("time:"):
                dt = float(ln.split()[1]) / 1000.0
                best = dt if best is None else min(best, dt)
    return best

def main():
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    exe, exe_jit = build()
    native = build_native()
    print("(in-process times; best of", RUNS, "runs; process startup excluded)")
    print(f"{'bench':<15}{'tree-walker':>13}{'vm':>10}{'jit':>10}{'python':>10}{'c -O2':>10}{'rust -O':>10}")
    for f, _desc in BENCHES:
        p = os.path.join("bench", f)
        tw = timed_wide([exe, p])
        vm = timed_wide([exe, "--vm", p])
        jt = timed_wide([exe_jit, p])
        py = timed_py(PY_EQUIV[f]) if f in PY_EQUIV else None
        nat = native.get(NATIVE.get(f, ""), {})
        c = timed_native(nat.get("c"))
        rs = timed_native(nat.get("rust"))
        fmt = lambda x: f"{x*1000:>7.1f} ms" if x is not None else "      n/a"
        print(f"{f:<15}{fmt(tw):>13}{fmt(vm):>10}{fmt(jt):>10}{fmt(py):>10}{fmt(c):>10}{fmt(rs):>10}")

if __name__ == "__main__":
    main()

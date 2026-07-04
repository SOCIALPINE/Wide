# The wide Programming Guide

wide is a language that **shows** costs and risks instead of hiding or forbidding them. Every run
prints your program's output followed by an illumination report — heap allocations, operation
costs, memory transfers, and borrow states appear as `INFO:` and `WARN:` lines woven next to the
source lines that caused them.

This document covers everything you need to write wide programs. Every example has been run and
verified.

## Contents

1. [Getting started](#1-getting-started)
2. [Values and variables](#2-values-and-variables)
3. [Operators](#3-operators)
4. [Control flow](#4-control-flow)
5. [Functions and closures](#5-functions-and-closures)
6. [Strings](#6-strings)
7. [Collections](#7-collections)
8. [Structs, enums, and pattern matching](#8-structs-enums-and-pattern-matching)
9. [Error handling](#9-error-handling)
10. [Modules](#10-modules)
11. [Input and output](#11-input-and-output)
12. [Memory and borrowing](#12-memory-and-borrowing)
13. [Tensors and automatic differentiation](#13-tensors-and-automatic-differentiation)
14. [Execution backends and performance](#14-execution-backends-and-performance)
15. [Cheat sheet](#15-cheat-sheet)

---

## 1. Getting started

```bash
cargo run -- program.wide          # run a program
cargo run -- --vm program.wide     # run on the bytecode VM
cargo test                         # the test suite
```

Execution always follows the same sequence: **static check → run → illumination report**. If the
static checker finds errors — undefined names, wrong argument counts, tensor shape mismatches,
definite borrow conflicts — the program does not run, and all errors are reported at once.

```
# comments start with #
print("hello, wide")
```

Statements are separated by newlines; there are no semicolons. Inside parentheses `(...)` and
brackets `[...]`, newlines count as whitespace, so multi-line literals and argument lists work
naturally.

## 2. Values and variables

```
n = 42                  # int
x = 3.14                # float (f32)
ok = true               # bool
s = "text"              # str
xs = [1, 2, 3]          # array
m = map{}               # map
r = 0..10               # range (half-open: 0 inclusive to 10 exclusive)
```

Variables are created by assignment. Types are dynamic; you may annotate them (`x: int = 5`) for
documentation — tensor shape annotations are the exception and are checked at compile time
(see chapter 13).

Arrays, maps, structs, and tensors have **reference semantics**: assigning them or passing them to
a function shares the same underlying data. Scalars (int, float, bool) and strings are copied by
value.

## 3. Operators

From lowest precedence to highest:

```
or  and  not             # logic (short-circuit, bool only)
== != < > <= >=          # comparison
..                       # range
+ -                      # addition/subtraction (+ also concatenates strings and arrays)
* /                      # multiplication/division
-x   not x   &x   *p     # unary
f(x)  x.m()  x.f  xs[i]  e?   # postfix
```

Integer division is integral; dividing by zero is a runtime error (with a warning illuminated).

## 4. Control flow

```
if x < 0 {
    print("negative")
} elif x == 0 {
    print("zero")
} else {
    print("positive")
}

while cond {
    ...
    break       # leave the loop
    continue    # next iteration
}

for i in 0..n { ... }       # over a range
for x in xs { ... }         # over an array
for ch in "abc" { ... }     # over a string, character by character
```

Conditions must be bool — zero and empty strings are not implicitly false.

While a `for` loop iterates an array, that array is shared-borrowed: mutating it inside the loop
(iterator invalidation) is a conflict and is blocked. Chapter 12 covers this in detail.

## 5. Functions and closures

```
fn add(a, b) {
    return a + b
}

fn fib(n) {                          # recursion works; so does mutual recursion,
    if n < 2 { return n }            # regardless of definition order
    return fib(n - 1) + fib(n - 2)
}
```

A function sees globals and its own parameters and locals — never the caller's locals.

Functions are also values:

```
add = fn(a, b) { return a + b }      # anonymous function
print(add(2, 3))                     # 5

d = add                              # assign a function value
fn apply(f, v) { return f(v) }       # take a function as an argument

fn make_adder(n) {                   # return a function — a closure capturing n
    return fn(x) { return x + n }
}
add5 = make_adder(5)
print(add5(100))                     # 105

xs = [1, 2, 3, 4]
print(xs.map(fn(x) { return x * x }))     # [1, 4, 9, 16] — a new array
print(xs.filter(fn(x) { return x > 2 }))  # [3, 4]
```

Closures capture **values at creation time**. Scalars are copied, so later reassignment is not
seen by the closure. Arrays, maps, and structs are shared references, so pushing to a captured
array inside a closure is visible outside.

## 6. Strings

```
s = "hello"
s.len                    # 5
s[1]                     # "e" (character indexing)
s[-1]                    # "o" (negative counts from the end)
s[1..4]                  # "ell" (slice — a copy)
s + ", world"            # concatenation

s.upper()  s.lower()  s.trim()
s.split(",")             # to an array
s.chars()                # array of characters
s.contains("ell")  s.starts_with("he")  s.ends_with("lo")
s.replace("l", "L")  s.find("ll")
```

Strings are immutable. When accumulating a string in a loop, use the mutable builder instead of
`s = s + c` (which copies every time, O(n²)):

```
b = strbuf()             # mutable string builder (amortized O(1) append)
b.push("a")
b.push("bc")
s = b.str()              # "abc"
b.clear()
```

## 7. Collections

### Arrays

Arrays are mutable and double as stacks, queues, and deques:

```
xs = [3, 1, 4]
xs[0]        xs[-1]        xs[1..3]         # read / from the end / slice (a copy)
xs[0] = 9                                   # write
xs.push(1)   xs.pop()                       # stack
xs.push_front(0)   xs.pop_front()           # queue / deque
xs.insert(1, 99)   xs.remove(0)
xs.sort()   xs.reverse()   xs.clear()
xs.len   xs.sum()   xs.contains(4)   xs.join(", ")
xs.map(f)   xs.filter(f)                    # higher-order methods (new arrays)
```

Slices are always **copies**, and out-of-range slices are errors (no silent clamping). Slices
cannot be assigned to.

### Maps

```
m = map{}
m["k"] = 1                    # insert or update
m["k"]                        # lookup (a missing key is an error)
m.get("k", 0)                 # lookup with a default
m.contains("k")   m.remove("k")
m.keys()   m.values()   m.len
```

Map keys may be int, str, or bool. Key order is deterministic (sorted).

### Heaps and sets

Priority queues and sets live in standard modules:

```
import "std/heap"
h = heap()
h.push(5)   h.push(1)
h.pop()                       # 1 (min-heap)
h.peek()

import "std/set"
s = set()
s.add(3)   s.contains(3)   s.remove(3)   s.items()
```

## 8. Structs, enums, and pattern matching

```
struct Point { x, y }

p = Point { x: 1, y: 2 }
p.x = 10                      # field assignment (reference semantics — visible to every sharer)

impl Point {
    fn dist2(self) { return self.x * self.x + self.y * self.y }
    fn shift(self, dx) { self.x = self.x + dx }    # mutating self is visible to the caller
}
p.dist2()
p.shift(3)
```

A method's first parameter is an explicit `self`.

```
enum Shape {
    Circle(r)
    Rect(w, h)
    Dot
}

s = Shape::Circle(5)

# statement match — arms are blocks
match s {
    Shape::Circle(r) => { print("circle", r) }
    Shape::Rect(w, h) => { print("rect", w * h) }
    Shape::Dot => { print("dot") }
}

# expression match — arms are expressions (separated by commas or newlines)
area = match s {
    Shape::Circle(r) => 3 * r * r,
    Shape::Rect(w, h) => w * h,
    _ => 0
}
```

Patterns include literals, the wildcard `_`, bindings, enum variants, and struct patterns
(`Point { x: 0, y }`). Enums may be recursive, so linked lists and trees can be defined directly.

## 9. Error handling

wide uses **error values**, not exceptions. A function returns either a normal value or an error
value, and the caller chooses how to handle it:

```
fn safe_div(a, b) {
    if b == 0 { return err("division by zero") }
    return a / b
}

# option 1: inspect directly
r = safe_div(10, 0)
if is_err(r) {
    print(err_msg(r))         # "division by zero"
}

# option 2: propagate with ? — on error, the current function returns that error
fn compute(a, b) {
    q = safe_div(a, b)?
    return q + 1
}
```

`?` bubbles an error up no matter how deep the chain. File I/O failures (chapter 11) come back as
the same kind of error value, so there is one uniform way to handle everything.

## 10. Modules

```
import "lib/util.wide"        # relative path
```

Functions, structs, and enums from an imported file are usable directly, and visibility is
transitive (indirect imports included). Duplicate and circular imports are handled safely.

Standard modules use the `std/` prefix: `import "std/ai"` (tensors), `import "std/fs"` (files),
`import "std/heap"`, `import "std/set"`. Using a gated feature without its import produces a
static error naming the import you need.

## 11. Input and output

### Console

```
print(a, b)                   # space-separated, newline appended

cout << "name: " << name << "\n"    # stream output — no automatic spacing, manual control
cin >> x >> y                       # input — splits on whitespace, auto-typed
                                    # (integer → int, decimal → float, otherwise → str)
```

`cin` targets may be variables, array elements (`xs[0]`), or fields (`p.x`).

### Files

```
import "std/fs"

write_file("notes.txt", "alpha\nbeta")    # create or overwrite
s = read_file("notes.txt")                # whole file as a string
lines = read_lines("notes.txt")           # array of lines
append_file("notes.txt", "\ngamma")
file_exists("notes.txt")                  # bool
remove_file("notes.txt")

r = read_file("missing.txt")              # failure is an error value — the program doesn't die
if is_err(r) { print(err_msg(r)) }
s = read_file(path)?                      # inside a function, propagate with ?
```

Every read and write illuminates its byte count — I/O cost is never hidden.

## 12. Memory and borrowing

wide's approach to memory access is one sentence: **don't block — show.**

### Pointers

```
xs = [10, 20, 30, 40]
p = &xs[1]             # address of an element — its provenance is illuminated
print(*p)              # 20
*p = 99                # write through the pointer (bounds-checked)
```

Out-of-bounds addresses (`&xs[9]`) are blocked, and a pointer left dangling by a shrunk array
refuses to dereference. The tracked record is always inspectable:

```
@show provenance p
# INFO: provenance: origin xs · extent 0..4 · pointee [1] · alive true · access owner
```

### Unchecked access — raw

Normal access blocks on violation. `raw.*` keeps the same tracking but only **illuminates** the
risk and proceeds — responsibility moves to the caller:

```
ps = &xs[0]
raw.read(ps, 3)          # INFO: within bounds — safe
raw.read(ps, 6)          # WARN: exceeds the extent — possible overrun (not blocked)
raw.write(ps, [1, 2])
raw.memcpy(dst, ps, 4)   # checked against both extents, illuminated
```

### Borrowing — a gradient of proof, guard, and trust

There is one borrow rule: **one write-borrow XOR many read-borrows.** What sets wide apart is
*how* the rule is enforced — as a continuum:

```
r = &xs                # shared borrow — mutating xs conflicts until this scope ends
m = &mut xs            # exclusive borrow — any other borrow or iteration conflicts
```

When the compiler can **prove** a borrow safe, there is no runtime guard at all (zero cost):

```
r = &xs
print(r[0])            # INFO: shared borrow of xs statically proven safe — cost 0
```

Definite conflicts are compile errors, caught before the program runs:

```
r = &xs
xs.push(2)             # compile error: borrow conflict (caught before run)
```

When a proof isn't possible (the variable escapes into a call, an alias appears, and so on), the
borrow automatically demotes to a **runtime guard** — and a potential conflict inside a
conditional is never a compile error, because it might never execute. wide does not reject
correct programs (zero false positives). Finally, to switch the checks off explicitly:

```
@trust xs.push(9)      # WARN: checks off for this one statement — the writer's responsibility
```

Where Rust offers "compile or give up", wide runs unprovable code anyway, guarded. Which tier
applied is always visible in the illumination.

## 13. Tensors and automatic differentiation

Enable with `import "std/ai"`. The design goal is a *see-through PyTorch*: every operation
illuminates its shape, bytes, FLOPs, transfers, and activation memory.

### Creating tensors and operating on them

```
import "std/ai"

a = tensor([[1, 2, 3], [4, 5, 6]])     # nested arrays → f32 tensor (shape inferred)
zeros([2, 3])   ones([2, 3])
a.shape   a.size   a.ndim

a + 1     a * 2                        # scalar broadcast
a + b                                  # elementwise (NumPy broadcasting rules)
matmul(a, w)                           # matrix multiply (2-D)
conv2d(x, k)                           # 2-D convolution (valid, stride 1)
maxpool2d(x, 2)                        # k×k max pooling
relu(t)  sigmoid(t)  tanh(t)  exp(t)  log(t)  softmax(t)  transpose(t)

a.sum()   a.mean()   a.max()           # reductions → scalar tensor (.item() for a number)
a.sum(0)  a.mean(1)                    # axis reductions
a.reshape([3, 2])                      # reshape (element count must match)
```

### Shapes are checked at compile time

Matmul dimension mismatches and impossible broadcasts are caught before the program runs.
Annotate symbolic dimensions and their *relationships* are checked too:

```
fn layer(x: tensor[(B, K)], w: tensor[(K, N)]) {
    return matmul(x, w)
}
# binding the two K's to different sizes at a call site is a compile error
```

Uppercase names are symbolic dimensions (relations checked), lowercase are dynamic, and `?`
means unknown.

### Autodiff and training

```
w = param([[0], [0]])          # trainable parameter (gradient tracked)
pred = matmul(x, w)
diff = pred - target
loss = (diff * diff).mean()    # the loss must be scalar to call backward
loss.backward()                # backpropagation
w.grad                         # accumulated gradient
grad_step(w, 0.01)             # one SGD step (resets the gradient)
adam_step(w, 0.01)             # one Adam step (moment state lives in the tensor)
```

This is enough to train regression models, multi-layer perceptrons, logistic and softmax
classifiers, and convolutional networks. Working examples live in `examples/ai/` — `cnn.wide`
differentiates a full conv2d → relu → maxpool2d → reshape → matmul chain and trains a
classifier end to end.

Current limits: `.backward()` requires a scalar, division is not differentiable, and matmul is
2-D only.

### GPU

```
cargo run --features gpu -- program.wide
```

With the `gpu` feature, matrix multiplication and elementwise operations on `.gpu()` tensors run
as real GPU compute shaders:

```
a = tensor([[...]]).gpu()      # a real upload — the adapter name appears in the illumination
b = tensor([[...]]).gpu()
c = matmul(a, b) * 2 + 1       # the whole chain stays on the device — zero re-uploads
print(c.cpu())
```

All transfers are illuminated. For small matrices the CPU can be faster than the GPU — the
transfer overhead dominates — and making that crossover visible is exactly what this language is
for. Measurements are in [BENCH.md](BENCH.md). Without the feature, the same code runs on the
CPU with identical results.

## 14. Execution backends and performance

| Backend | How to run | Notes |
|---------|-----------|-------|
| Tree-walker | `cargo run -- f.wide` | Default; supports everything (the reference) |
| Bytecode VM | `--vm` | The full core language; closures, borrows, and tensors are tree-walker-only for now |
| JIT | `--features jit` | Numeric functions (int/f64, recursion included) compile to machine code |
| GPU | `--features gpu` | Tensor matmul and elementwise ops as GPU shaders |

The JIT selects eligible functions automatically (visible in the illumination); everything else
runs interpreted — no code changes required. On fib(30) it is several hundred times faster than
the interpreter. Exact numbers and methodology are in [BENCH.md](BENCH.md).

The `--time` flag prints the execution time to stderr (excluding process startup — for
benchmarking).

## 15. Cheat sheet

```
# variables       x = 5      x: int = 5
# output          print(a, b)       cout << a << "\n"
# input           cin >> x >> y                       # auto-typed
# arith/compare   + - * /   == != < > <= >=
# logic           and  or  not                        # short-circuit
# range/index     0..n   xs[i]   xs[-1]   xs[1..3]    # slices are copies
# conditionals    if c { } elif d { } else { }
# loops           while c { }   for x in xs { }   break   continue
# functions       fn f(a, b) { return a + b }
# closures        g = fn(x) { return x + k }   xs.map(f)   xs.filter(f)
# strings         "..."  .upper() .split(s) .len       # accumulate with strbuf()
# arrays          [1, 2]  .push .pop .sort .sum .len
# maps            map{}   m[k] = v   .get(k, default) .keys
# structs         struct P { x, y }   P { x: 1, y: 2 }   p.x
# methods         impl P { fn m(self) { ... } }   p.m()
# enums           enum E { A(v)  B }   E::A(5)
# matching        match v { E::A(x) => { ... }  _ => { ... } }
# errors          err(m)  is_err(v)  err_msg(v)  f()?
# modules         import "file.wide"   import "std/ai" "std/fs" "std/heap" "std/set"
# files           read_file  read_lines  write_file  append_file  remove_file  file_exists
# pointers        p = &xs[i]   *p   @show provenance p
# borrows         r = &xs   m = &mut xs   @trust <stmt>
# raw             raw.read(p, n)  raw.write(p, vals)  raw.memcpy(d, s, n)
# tensors         tensor  param  zeros  ones  matmul  conv2d  maxpool2d  relu  softmax
# autodiff        loss.backward()   w.grad   grad_step(w, lr)   adam_step(w, lr)
```

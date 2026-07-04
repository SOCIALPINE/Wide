//! wide v0.2 integration tests. Verifies the full base grammar.

use std::path::Path;

use wide::{eval_program, Value};

/// Helper that runs a source and fetches a top-level variable's value.
fn val(src: &str, var: &str) -> Value {
    eval_program(src)
        .unwrap_or_else(|e| panic!("execution failed: {}\n--- source ---\n{}", e, src))
        .get(var)
        .unwrap_or_else(|| panic!("variable '{}' not found", var))
}

fn int(n: i64) -> Value {
    Value::Int(n)
}
fn boolean(b: bool) -> Value {
    Value::Bool(b)
}
fn string(s: &str) -> Value {
    Value::Str(s.to_string())
}

#[test]
fn arithmetic_and_precedence() {
    assert_eq!(val("x = 2 + 3 * 4", "x"), int(14));
    assert_eq!(val("x = (2 + 3) * 4", "x"), int(20));
    assert_eq!(val("x = 7 / 2", "x"), int(3));
    assert_eq!(val("x = -5 + 2", "x"), int(-3));
    assert_eq!(val("x = 10 - 2 - 3", "x"), int(5)); // left-associative
}

#[test]
fn float_promotion() {
    assert_eq!(val("x = 1 / 2.0", "x"), Value::Float(0.5));
    assert_eq!(val("x = 3 + 0.5", "x"), Value::Float(3.5));
}

#[test]
fn comparisons() {
    assert_eq!(val("x = 3 < 5", "x"), boolean(true));
    assert_eq!(val("x = 5 < 3", "x"), boolean(false));
    assert_eq!(val("x = 3 == 3", "x"), boolean(true));
    assert_eq!(val("x = 3 != 4", "x"), boolean(true));
    assert_eq!(val("x = 5 <= 5", "x"), boolean(true));
    assert_eq!(val("x = 5 >= 6", "x"), boolean(false));
    assert_eq!(val(r#"x = "a" < "b""#, "x"), boolean(true));
}

#[test]
fn logical_and_short_circuit() {
    assert_eq!(val("x = true and false", "x"), boolean(false));
    assert_eq!(val("x = true or false", "x"), boolean(true));
    assert_eq!(val("x = not false", "x"), boolean(true));
    // Short-circuit: the right side (divide by zero) would error if evaluated, so no panic means short-circuit worked.
    assert_eq!(val("x = false and (1 / 0 == 0)", "x"), boolean(false));
    assert_eq!(val("x = true or (1 / 0 == 0)", "x"), boolean(true));
}

#[test]
fn if_elif_else() {
    let src = r#"
fn classify(n) {
    if n < 0 { return "neg" }
    elif n == 0 { return "zero" }
    else { return "pos" }
}
a = classify(-3)
b = classify(0)
c = classify(7)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), string("neg"));
    assert_eq!(it.get("b").unwrap(), string("zero"));
    assert_eq!(it.get("c").unwrap(), string("pos"));
}

#[test]
fn while_loop() {
    let src = r#"
n = 0
i = 1
while i <= 5 {
    n = n + i
    i = i + 1
}
"#;
    assert_eq!(val(src, "n"), int(15));
}

#[test]
fn for_over_range_and_array() {
    let range = r#"
total = 0
for i in 1..5 {
    total = total + i
}
"#;
    assert_eq!(val(range, "total"), int(10)); // 1+2+3+4

    let array = r#"
xs = [10, 20, 30]
total = 0
for x in xs {
    total = total + x
}
"#;
    assert_eq!(val(array, "total"), int(60));
}

#[test]
fn break_and_continue() {
    let brk = r#"
sum = 0
for i in 0..100 {
    if i == 5 { break }
    sum = sum + i
}
"#;
    assert_eq!(val(brk, "sum"), int(10)); // 0+1+2+3+4

    let cont = r#"
sum = 0
for i in 0..10 {
    if i == 5 { continue }
    sum = sum + i
}
"#;
    assert_eq!(val(cont, "sum"), int(40)); // 45 - 5
}

#[test]
fn recursion_fib() {
    let src = r#"
fn fib(n) {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}
x = fib(10)
"#;
    assert_eq!(val(src, "x"), int(55));
}

#[test]
fn mutual_recursion() {
    let src = r#"
fn is_even(n) {
    if n == 0 { return true }
    return is_odd(n - 1)
}
fn is_odd(n) {
    if n == 0 { return false }
    return is_even(n - 1)
}
a = is_even(10)
b = is_odd(7)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), boolean(true));
    assert_eq!(it.get("b").unwrap(), boolean(true));
}

#[test]
fn functions_see_globals_not_caller_locals() {
    // Globals are visible.
    let src = r#"
g = 100
fn f() { return g + 1 }
x = f()
"#;
    assert_eq!(val(src, "x"), int(101));

    // Function arguments don't leak outside.
    assert!(eval_program("fn f(a) { return a }\nr = f(5)\ny = a").is_err());
}

#[test]
fn strings() {
    assert_eq!(val(r#"s = "hello" + " " + "world""#, "s"), string("hello world"));
    assert_eq!(val(r#"n = "héllo".len"#, "n"), int(5));
}

#[test]
fn arrays_sum_len() {
    assert_eq!(val("x = [3, 1, 4, 1, 5].sum()", "x"), int(14));
    assert_eq!(val("x = [3, 1, 4, 1, 5].len", "x"), int(5));
}

#[test]
fn reassignment_updates_outer() {
    let src = r#"
x = 1
if true {
    x = 2
}
"#;
    assert_eq!(val(src, "x"), int(2));
}

#[test]
fn illumination_channel_records_costs() {
    let it = eval_program("m = map{}\nxs = [1, 2, 3]\nh = heap()").unwrap();
    let msgs: Vec<&str> = it.channel.records.iter().map(|r| r.msg.as_str()).collect();
    // Mutable collections all live on the heap — illumination honestly shows that cost.
    assert!(msgs.iter().filter(|m| m.contains("heap")).count() >= 3, "need 3 heap illuminations: {:?}", msgs);
}

fn arr(xs: Vec<Value>) -> Value {
    Value::array(xs)
}

#[test]
fn indexing_get_and_set() {
    assert_eq!(val("xs = [10, 20, 30]\nx = xs[1]", "x"), int(20));
    assert_eq!(val("s = \"hello\"\nc = s[1]", "c"), string("e"));
    let src = "xs = [1, 2, 3]\nxs[1] = 99";
    assert_eq!(val(src, "xs"), arr(vec![int(1), int(99), int(3)]));
}

#[test]
fn stack_via_vector() {
    let src = r#"
st = []
st.push(1)
st.push(2)
st.push(3)
a = st.pop()
b = st.pop()
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(3)); // LIFO
    assert_eq!(it.get("b").unwrap(), int(2));
    assert_eq!(it.get("st").unwrap(), arr(vec![int(1)]));
}

#[test]
fn queue_via_vector() {
    let src = r#"
q = []
q.push(1)
q.push(2)
q.push(3)
a = q.pop_front()
b = q.pop_front()
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(1)); // FIFO
    assert_eq!(it.get("b").unwrap(), int(2));
}

#[test]
fn min_heap() {
    let src = r#"
h = heap()
h.push(5)
h.push(1)
h.push(3)
h.push(2)
a = h.pop()
b = h.pop()
c = h.pop()
d = h.pop()
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(1));
    assert_eq!(it.get("b").unwrap(), int(2));
    assert_eq!(it.get("c").unwrap(), int(3));
    assert_eq!(it.get("d").unwrap(), int(5));
}

#[test]
fn vector_methods() {
    assert_eq!(val("xs = [3, 1, 2]\nxs.sort()", "xs"), arr(vec![int(1), int(2), int(3)]));
    assert_eq!(val("xs = [1, 2, 3]\nxs.reverse()", "xs"), arr(vec![int(3), int(2), int(1)]));
    assert_eq!(val("xs = [1, 2, 3]\nxs.insert(1, 9)", "xs"), arr(vec![int(1), int(9), int(2), int(3)]));
    assert_eq!(val("xs = [1, 2, 3]\nr = xs.remove(0)", "r"), int(1));
    assert_eq!(val("x = [1, 2, 3].contains(2)", "x"), boolean(true));
    assert_eq!(val(r#"s = [1, 2, 3].join("-")"#, "s"), string("1-2-3"));
    assert_eq!(val("n = [1, 2, 3].len", "n"), int(3));
}

#[test]
fn map_operations() {
    let src = r#"
m = map{}
m["a"] = 1
m["b"] = 2
x = m["a"]
has = m.contains("b")
nokey = m.contains("z")
n = m.keys().len
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("x").unwrap(), int(1));
    assert_eq!(it.get("has").unwrap(), boolean(true));
    assert_eq!(it.get("nokey").unwrap(), boolean(false));
    assert_eq!(it.get("n").unwrap(), int(2));
    assert_eq!(val(r#"m = map{}
m[1] = 10
r = m.get(2, -1)"#, "r"), int(-1)); // default value
}

#[test]
fn reference_semantics_mutation_through_function() {
    // Passing an array to a function shares the same Rc, so pushing inside is visible outside.
    let src = r#"
fn fill(xs) {
    xs.push(99)
}
ys = [1]
fill(ys)
n = ys.len
"#;
    assert_eq!(val(src, "n"), int(2));
}

#[test]
fn builtins_numeric_and_convert() {
    assert_eq!(val("x = len([1, 2, 3, 4])", "x"), int(4));
    assert_eq!(val("x = abs(-7)", "x"), int(7));
    assert_eq!(val("x = min(3, 1, 2)", "x"), int(1));
    assert_eq!(val("x = max(3, 1, 2)", "x"), int(3));
    assert_eq!(val("x = pow(2, 10)", "x"), int(1024));
    assert_eq!(val("x = sqrt(9.0)", "x"), Value::Float(3.0));
    assert_eq!(val("x = floor(3.7)", "x"), int(3));
    assert_eq!(val("x = ceil(3.2)", "x"), int(4));
    assert_eq!(val("x = int(3.9)", "x"), int(3));
    assert_eq!(val(r#"x = int("42")"#, "x"), int(42));
    assert_eq!(val(r#"x = str(42)"#, "x"), string("42"));
    assert_eq!(val(r#"x = float("3.5")"#, "x"), Value::Float(3.5));
}

#[test]
fn builtins_hex_bin_char() {
    assert_eq!(val("x = hex(255)", "x"), string("0xff"));
    assert_eq!(val("x = bin(5)", "x"), string("0b101"));
    assert_eq!(val("x = hex(-255)", "x"), string("-0xff"));
    assert_eq!(val(r#"x = int("ff", 16)"#, "x"), int(255));
    assert_eq!(val(r#"x = int("1010", 2)"#, "x"), int(10));
    assert_eq!(val(r#"x = ord("A")"#, "x"), int(65));
    assert_eq!(val("x = chr(65)", "x"), string("A"));
}

#[test]
fn string_methods() {
    assert_eq!(val(r#"x = "Hello".upper()"#, "x"), string("HELLO"));
    assert_eq!(val(r#"x = "Hello".lower()"#, "x"), string("hello"));
    assert_eq!(val(r#"x = "  hi  ".trim()"#, "x"), string("hi"));
    assert_eq!(val(r#"x = "a,b,c".split(",")"#, "x"), arr(vec![string("a"), string("b"), string("c")]));
    assert_eq!(val(r#"x = "hello".contains("ell")"#, "x"), boolean(true));
    assert_eq!(val(r#"x = "hello".starts_with("he")"#, "x"), boolean(true));
    assert_eq!(val(r#"x = "hello".ends_with("lo")"#, "x"), boolean(true));
    assert_eq!(val(r#"x = "hello".find("l")"#, "x"), int(2));
    assert_eq!(val(r#"x = "hello".replace("l", "L")"#, "x"), string("heLLo"));
}

#[test]
fn assert_builtin() {
    assert!(eval_program("assert(1 == 1)").is_ok());
    assert!(eval_program("assert(1 == 2)").is_err());
    assert!(eval_program(r#"assert(false, "boom")"#).is_err());
}

#[test]
fn algorithm_heapsort() {
    // Sort via heap: push everything, then popping yields ascending order.
    let src = r#"
fn heapsort(xs) {
    h = heap()
    for x in xs {
        h.push(x)
    }
    out = []
    while h.len > 0 {
        out.push(h.pop())
    }
    return out
}
result = heapsort([5, 3, 8, 1, 9, 2])
"#;
    assert_eq!(
        val(src, "result"),
        arr(vec![int(1), int(2), int(3), int(5), int(8), int(9)])
    );
}

#[test]
fn algorithm_word_count() {
    // Count word frequencies with a map.
    let src = r#"
fn count_words(text) {
    counts = map{}
    for w in text.split(" ") {
        if counts.contains(w) {
            counts[w] = counts[w] + 1
        } else {
            counts[w] = 1
        }
    }
    return counts
}
m = count_words("a b a c a b")
a = m["a"]
b = m["b"]
c = m["c"]
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(3));
    assert_eq!(it.get("b").unwrap(), int(2));
    assert_eq!(it.get("c").unwrap(), int(1));
}

#[test]
fn struct_basics_and_mutation() {
    let src = r#"
struct Point { x, y }
p = Point { x: 3, y: 4 }
a = p.x
p.x = 10
b = p.x
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(3));
    assert_eq!(it.get("b").unwrap(), int(10));
}

#[test]
fn struct_reference_through_function() {
    let src = r#"
struct Box { v }
fn bump(b) { b.v = b.v + 1 }
bx = Box { v: 10 }
bump(bx)
bump(bx)
x = bx.v
"#;
    assert_eq!(val(src, "x"), int(12));
}

#[test]
fn enum_and_match_with_payload() {
    let src = r#"
enum Shape { Circle(r) Rect(w, h) Dot }
fn area(s) {
    match s {
        Shape::Circle(r) => { return 3 * r * r }
        Shape::Rect(w, h) => { return w * h }
        Shape::Dot => { return 0 }
    }
}
a = area(Shape::Circle(10))
b = area(Shape::Rect(3, 4))
c = area(Shape::Dot)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(300));
    assert_eq!(it.get("b").unwrap(), int(12));
    assert_eq!(it.get("c").unwrap(), int(0));
}

#[test]
fn match_literals_and_wildcard() {
    let src = r#"
fn classify(n) {
    match n {
        0 => "zero"
        1 => "one"
        _ => "many"
    }
}
"#;
    // Single-expression arms produce a value, but a statement match needs a return to capture it — there's no function here,
    // so verify separately that literal/wildcard matching picks the correct arm.
    let src2 = r#"
out = []
for n in [0, 1, 5, 1] {
    match n {
        0 => { out.push("zero") }
        1 => { out.push("one") }
        _ => { out.push("many") }
    }
}
"#;
    let _ = src; // (just to confirm single-expression arm syntax parses)
    assert!(eval_program(src).is_ok());
    let it = eval_program(src2).unwrap();
    assert_eq!(
        it.get("out").unwrap(),
        arr(vec![string("zero"), string("one"), string("many"), string("one")])
    );
}

#[test]
fn match_struct_pattern() {
    let src = r#"
struct Point { x, y }
fn describe(p) {
    match p {
        Point { x: 0, y: 0 } => { return "origin" }
        Point { x, y } => { return x + y }
    }
}
a = describe(Point { x: 0, y: 0 })
b = describe(Point { x: 2, y: 3 })
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), string("origin"));
    assert_eq!(it.get("b").unwrap(), int(5));
}

#[test]
fn recursive_enum_list_sum() {
    let src = r#"
enum List { Cons(head, tail) Nil }
fn sum(lst) {
    match lst {
        List::Cons(h, t) => { return h + sum(t) }
        List::Nil => { return 0 }
    }
}
total = sum(List::Cons(1, List::Cons(2, List::Cons(3, List::Nil))))
"#;
    assert_eq!(val(src, "total"), int(6));
}

#[test]
fn ast_evaluator_bootstrap_miniature() {
    // Mini AST interpreter — a miniature of bootstrapping (the compiler written in wide).
    let src = r#"
enum Expr { Num(n) Add(a, b) Mul(a, b) }
fn eval(e) {
    match e {
        Expr::Num(n) => { return n }
        Expr::Add(a, b) => { return eval(a) + eval(b) }
        Expr::Mul(a, b) => { return eval(a) * eval(b) }
    }
}
tree = Expr::Mul(Expr::Add(Expr::Num(2), Expr::Num(3)), Expr::Num(4))
result = eval(tree)
"#;
    assert_eq!(val(src, "result"), int(20)); // (2+3)*4
}

#[test]
fn match_no_arm_and_def_errors() {
    assert!(eval_program("enum E { A B }\nmatch E::A {\n  E::B => { x = 1 }\n}").is_err(), "no matching arm");
    assert!(eval_program("struct P { x }\np = P { y: 1 }").is_err(), "unknown field");
    assert!(eval_program("struct P { x, y }\np = P { x: 1 }").is_err(), "missing field");
    assert!(eval_program("enum E { A(n) }\ne = E::A(1, 2)").is_err(), "variant arg count mismatch");
}

#[test]
fn module_import_transitive_and_dedup() {
    // app → shapes → mathx, and app → mathx (a diamond). mathx is loaded only once (no cycles).
    let prog = wide::load_file(Path::new("tests/modules/app.wide")).unwrap();
    let mut it = wide::Interp::new();
    it.run(&prog).unwrap();
    assert_eq!(it.get("area").unwrap(), int(75)); // circle_area(Circle{r:5}) = 3*25
    assert_eq!(it.get("cubed").unwrap(), int(27)); // mathx's cube, visible transitively
}

#[test]
fn module_missing_file_errors() {
    assert!(wide::load_file(Path::new("tests/modules/nope.wide")).is_err());
}

#[test]
fn unresolved_import_in_eval_program_errors() {
    // eval_program (single source) has no import resolution, so this is a runtime error.
    assert!(eval_program(r#"import "x.wide""#).is_err());
}

#[cfg(feature = "ai")]
#[test]
fn tensor_creation_and_reductions() {
    assert_eq!(val("a = tensor([[1,2,3],[4,5,6]])\ns = a.size", "s"), int(6));
    assert_eq!(val("a = tensor([[1,2,3],[4,5,6]])\nn = a.ndim", "n"), int(2));
    assert_eq!(val("a = tensor([1,2,3,4])\nx = a.sum().item()", "x"), Value::Float(10.0));
    assert_eq!(val("a = tensor([2,4,6])\nx = a.mean().item()", "x"), Value::Float(4.0));
    assert_eq!(val("a = tensor([[1,2],[3,4]])\nsh = a.shape", "sh"), arr(vec![int(2), int(2)]));
}

#[cfg(feature = "ai")]
#[test]
fn tensor_elementwise_and_broadcast() {
    assert_eq!(val("a = tensor([1,2,3])\nx = (a * 2).sum().item()", "x"), Value::Float(12.0));
    assert_eq!(val("a = tensor([1,2,3])\nx = (a + 10).sum().item()", "x"), Value::Float(36.0));
    assert_eq!(
        val("a = tensor([1,2,3])\nb = tensor([10,20,30])\nx = (a + b).sum().item()", "x"),
        Value::Float(66.0)
    );
}

#[cfg(feature = "ai")]
#[test]
fn tensor_matmul() {
    // [[1,2,3],[4,5,6]] · [[1,0],[0,1],[1,1]] = [[4,5],[10,11]], sum = 30
    let src = "a = tensor([[1,2,3],[4,5,6]])\nw = tensor([[1,0],[0,1],[1,1]])\nx = matmul(a, w).sum().item()";
    assert_eq!(val(src, "x"), Value::Float(30.0));
}

#[cfg(feature = "ai")]
#[test]
fn tensor_shape_errors_and_cost_illumination() {
    assert!(eval_program("a = tensor([[1,2,3]])\nb = matmul(a, a)").is_err(), "matmul dimension mismatch");
    assert!(eval_program("a = tensor([1,2,3])\nb = tensor([1,2])\nc = a + b").is_err(), "elementwise shape mismatch");
    // Whether cost (FLOPs) is recorded in illumination — wide's identity (visible-cost).
    let it = eval_program("a = tensor([[1,2],[3,4]])\nb = matmul(a, a)").unwrap();
    let msgs: Vec<&str> = it.channel.records.iter().map(|r| r.msg.as_str()).collect();
    assert!(msgs.iter().any(|m| m.contains("FLOP")), "matmul FLOP illumination required: {:?}", msgs);
    assert!(msgs.iter().any(|m| m.contains("tensor f32")), "tensor-creation illumination required: {:?}", msgs);
}

#[cfg(feature = "ai")]
#[test]
fn autodiff_linear_regression_gradient() {
    // Gradient of the loss wrt w = 2·xᵀ@(x@w - target). Check it matches the hand-computed value.
    let src = r#"
x = tensor([[1, 2], [3, 4]])
w = param([[1], [1]])
target = tensor([[5], [11]])
pred = matmul(x, w)
diff = pred - target
loss = (diff * diff).sum()
loss.backward()
g = w.grad
l = loss.item()
g0 = g.shape
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("l").unwrap(), Value::Float(20.0)); // (-2)²+(-4)² = 20
    // w.grad = [-28, -40] — verified via sum
    assert_eq!(val(&format!("{}\ngs = w.grad.sum().item()", src), "gs"), Value::Float(-68.0));
}

// ---- v0.34: AI depth — axis-wise reduction (sum/mean over an axis, differentiable) ----

#[cfg(feature = "ai")]
#[test]
fn axis_reduction_forward() {
    // x = [[1,2,3],[4,5,6]]: sum(0)=[5,7,9], sum(1)=[6,15], mean(0)=[2.5,3.5,4.5], mean(1)=[2,5].
    let x = "x = tensor([[1, 2, 3], [4, 5, 6]])\n";
    assert_eq!(val(&format!("{}s = x.sum(0).sum().item()", x), "s"), Value::Float(21.0));
    assert_eq!(val(&format!("{}s = x.sum(1).sum().item()", x), "s"), Value::Float(21.0));
    assert_eq!(val(&format!("{}s = x.mean(0).sum().item()", x), "s"), Value::Float(10.5));
    assert_eq!(val(&format!("{}s = x.mean(1).sum().item()", x), "s"), Value::Float(7.0));
}

#[cfg(feature = "ai")]
#[test]
fn axis_reduction_backward() {
    // sum over axis: each input contributes once → grad all 1 (6 elems → sum 6).
    let s = "w = param([[1, 1, 1], [1, 1, 1]])\nl = w.sum(0).sum()\nl.backward()\ng = w.grad.sum().item()";
    assert_eq!(val(s, "g"), Value::Float(6.0));
    // mean over axis 0 (rows=2) → each grad 1/2 (6 elems → sum 3).
    let m = "w = param([[1, 1, 1], [1, 1, 1]])\nl = w.mean(0).sum()\nl.backward()\ng = w.grad.sum().item()";
    assert_eq!(val(m, "g"), Value::Float(3.0));
}

#[cfg(feature = "ai")]
#[test]
fn axis_reduction_errors() {
    assert!(eval_program("x = tensor([[1, 2], [3, 4]])\ny = x.sum(2)").is_err(), "axis out of range");
    assert!(eval_program("x = tensor([1, 2, 3])\ny = x.sum(0)").is_err(), "axis reduce needs 2D");
}

// ---- v0.38: AI depth — reshape (differentiable view) ----

#[cfg(feature = "ai")]
#[test]
fn reshape_forward_and_total_check() {
    // (2,3) → (3,2) keeps the data; reduce to confirm contents survive; element-count mismatch errors.
    assert_eq!(val("t = tensor([[1, 2, 3], [4, 5, 6]])\ns = t.reshape([3, 2]).sum().item()", "s"), Value::Float(21.0));
    assert_eq!(val("t = tensor([[1, 2, 3], [4, 5, 6]])\ns = t.reshape([6]).sum().item()", "s"), Value::Float(21.0));
    assert!(eval_program("t = tensor([[1, 2, 3], [4, 5, 6]])\nx = t.reshape([2, 2])").is_err(), "element count must match");
}

#[cfg(feature = "ai")]
#[test]
fn reshape_is_differentiable() {
    // flatten then sum → grad flows back unchanged (all ones), 4 elements → sum 4.
    let s = "w = param([[1, 1], [1, 1]])\nl = w.reshape([4]).sum()\nl.backward()\ng = w.grad.sum().item()";
    assert_eq!(val(s, "g"), Value::Float(4.0));
}

// ---- v0.39: AI depth — conv2d (valid 2D cross-correlation, differentiable) ----

#[cfg(feature = "ai")]
#[test]
fn conv2d_forward() {
    // k = [[1,0],[0,1]] picks x[i,j] + x[i+1,j+1]: out = [[6,8],[12,14]], sum 40, shape (2,2).
    let src = "x = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\nk = tensor([[1, 0], [0, 1]])\ny = conv2d(x, k)\ns = y.sum().item()\nsh = y.shape";
    assert_eq!(val(src, "s"), Value::Float(40.0));
    assert_eq!(val(src, "sh"), arr(vec![int(2), int(2)]));
}

#[cfg(feature = "ai")]
#[test]
fn conv2d_is_differentiable() {
    // 3x3 input ⋆ 2x2 kernel, all ones, loss = out.sum().
    // dX[a,b] = #windows covering (a,b) → total 2·2·2·2 = 16. dK[p,q] = Σ g·x = 4 each → total 16.
    let x = "x = param([[1, 1, 1], [1, 1, 1], [1, 1, 1]])\nk = tensor([[1, 1], [1, 1]])\nl = conv2d(x, k).sum()\nl.backward()\ng = x.grad.sum().item()";
    assert_eq!(val(x, "g"), Value::Float(16.0));
    let k = "x = tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])\nk = param([[1, 1], [1, 1]])\nl = conv2d(x, k).sum()\nl.backward()\ng = k.grad.sum().item()";
    assert_eq!(val(k, "g"), Value::Float(16.0));
}

#[cfg(feature = "ai")]
#[test]
fn conv2d_errors() {
    assert!(eval_program("x = tensor([[1, 2], [3, 4]])\ny = conv2d(x, tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))").is_err(), "kernel larger than input");
    assert!(eval_program("x = tensor([1, 2, 3])\ny = conv2d(x, x)").is_err(), "conv2d needs 2D tensors");
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_checks_conv2d() {
    // Kernel larger than input — caught before run.
    let errs = wide::type_errors("import \"std/ai\"\nc = conv2d(zeros([2, 2]), zeros([3, 3]))").unwrap();
    assert!(errs.iter().any(|e| e.contains("conv2d kernel")), "kernel-too-big caught statically: {:?}", errs);
    // Inference flows through conv2d: output (5−3+1, 5−3+1) = (3,3); matmul inner 3≠4 is then caught.
    let errs2 = wide::type_errors("import \"std/ai\"\nc = matmul(conv2d(zeros([5, 5]), zeros([3, 3])), zeros([4, 7]))").unwrap();
    assert!(errs2.iter().any(|e| e.contains("matmul dimension mismatch")), "conv2d output shape inferred: {:?}", errs2);
    // Valid programs pass (no false positives).
    assert!(wide::type_errors("import \"std/ai\"\nc = matmul(conv2d(zeros([5, 5]), zeros([3, 3])), zeros([3, 7]))").unwrap().is_empty());
}

// ---- v0.40: AI depth — maxpool2d (non-overlapping max pooling, differentiable) + CNN pipeline ----

#[cfg(feature = "ai")]
#[test]
fn maxpool2d_forward() {
    // [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] pooled 2×2 → [[6,8],[14,16]], sum 44.
    let src = "x = tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])\ny = maxpool2d(x, 2)\ns = y.sum().item()\nsh = y.shape";
    assert_eq!(val(src, "s"), Value::Float(44.0));
    assert_eq!(val(src, "sh"), arr(vec![int(2), int(2)]));
    // Trailing rows/cols that don't fill a window are dropped: (3,3) pooled by 2 → (1,1) keeping max of the top-left window.
    assert_eq!(val("x = tensor([[9, 2, 3], [4, 5, 6], [7, 8, 1]])\ns = maxpool2d(x, 2).sum().item()", "s"), Value::Float(9.0));
}

#[cfg(feature = "ai")]
#[test]
fn maxpool2d_backward_routes_to_argmax() {
    // Gradient flows only to each window's max element: 4 windows → grad sum 4; non-max entries get 0.
    let s = "w = param([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])\nl = maxpool2d(w, 2).sum()\nl.backward()\ng = w.grad.sum().item()";
    assert_eq!(val(s, "g"), Value::Float(4.0));
    // The max of the whole 2×2 (single window) is one element — its grad is 1, so grad picks it out.
    let m = "w = param([[1, 9], [3, 4]])\nl = maxpool2d(w, 2).sum()\nl.backward()\ng = (w.grad * w).sum().item()";
    assert_eq!(val(m, "g"), Value::Float(9.0));
}

#[cfg(feature = "ai")]
#[test]
fn maxpool2d_errors() {
    assert!(eval_program("x = tensor([[1, 2], [3, 4]])\ny = maxpool2d(x, 3)").is_err(), "window larger than input");
    assert!(eval_program("x = tensor([1, 2, 3])\ny = maxpool2d(x, 2)").is_err(), "needs 2D tensor");
    assert!(eval_program("x = tensor([[1, 2], [3, 4]])\ny = maxpool2d(x, 0)").is_err(), "k must be >= 1");
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_checks_maxpool2d() {
    // Window larger than input — caught before run.
    let errs = wide::type_errors("import \"std/ai\"\nc = maxpool2d(zeros([2, 2]), 3)").unwrap();
    assert!(errs.iter().any(|e| e.contains("maxpool2d window")), "window-too-big caught statically: {:?}", errs);
    // Inference flows: (8,8) pooled by 2 → (4,4); matmul inner 4≠5 is then caught.
    let errs2 = wide::type_errors("import \"std/ai\"\nc = matmul(maxpool2d(zeros([8, 8]), 2), zeros([5, 7]))").unwrap();
    assert!(errs2.iter().any(|e| e.contains("matmul dimension mismatch")), "maxpool2d output shape inferred: {:?}", errs2);
    assert!(wide::type_errors("import \"std/ai\"\nc = matmul(maxpool2d(zeros([8, 8]), 2), zeros([4, 7]))").unwrap().is_empty());
}

#[cfg(feature = "ai")]
#[test]
fn cnn_pipeline_trains() {
    // Full CNN chain — conv2d → relu → maxpool2d → reshape (flatten) → matmul (dense) — trained end to
    // end with Adam to separate a vertical-stripe image from a horizontal-stripe one (BCE loss).
    let src = r#"
v = tensor([[0, 9, 0, 9, 0], [0, 9, 0, 9, 0], [0, 9, 0, 9, 0], [0, 9, 0, 9, 0], [0, 9, 0, 9, 0]])
h = tensor([[0, 0, 0, 0, 0], [9, 9, 9, 9, 9], [0, 0, 0, 0, 0], [9, 9, 9, 9, 9], [0, 0, 0, 0, 0]])
k = param([[0.1, 0], [0, 0.1]])
w = param([[0.1], [0.1], [0.1], [0.1]])
b = param([[0]])
fn logit(img, k, w, b) {
    f = maxpool2d(relu(conv2d(img, k)), 2)
    return matmul(f.reshape([1, 4]), w) + b
}
step = 0
loss = tensor([0])
while step < 150 {
    pv = sigmoid(logit(v, k, w, b))
    ph = sigmoid(logit(h, k, w, b))
    loss = ((tensor([[1]]) - pv) * (tensor([[1]]) - pv)).sum() + (ph * ph).sum()
    loss.backward()
    adam_step(k, 0.05)
    adam_step(w, 0.05)
    adam_step(b, 0.05)
    step = step + 1
}
l = loss.item()
"#;
    let it = eval_program(src).unwrap();
    let l = match it.get("l").unwrap() {
        Value::Float(x) => x,
        other => panic!("loss should be a float, got {:?}", other),
    };
    assert!(l < 0.01, "CNN should converge (final loss {})", l);
}

// ---- v0.35: AI depth — Adam optimizer (per-parameter moment state in the tensor) ----

#[cfg(feature = "ai")]
#[test]
fn adam_optimizer_converges() {
    // Fit w so x·w ≈ target (true w = [1; 2]). Adam's moment state persists across steps in the tensor.
    let src = r#"
x = tensor([[1, 2], [3, 4], [5, 6]])
target = tensor([[5], [11], [17]])
w = param([[0], [0]])
i = 0
while i < 400 {
    diff = matmul(x, w) - target
    loss = (diff * diff).mean()
    loss.backward()
    adam_step(w, 0.3)
    i = i + 1
}
d = matmul(x, w) - target
fl = (d * d).mean().item()
"#;
    let it = eval_program(src).unwrap();
    if let Value::Float(loss) = it.get("fl").unwrap() {
        assert!(loss < 0.01, "Adam should converge (final loss {})", loss);
    } else {
        panic!("fl is not a float");
    }
}

#[cfg(feature = "ai")]
#[test]
fn adam_step_requires_grad() {
    assert!(eval_program("w = param([[1], [1]])\nadam_step(w, 0.1)").is_err(), "adam_step needs backward first");
}

// ---- v0.37: Cranelift JIT (stage 3 — real native code, integer subset, `jit` feature) ----

#[cfg(feature = "jit")]
#[test]
fn jit_runs_native_and_matches() {
    // An integer loop function is JIT-compiled and called natively; the result matches the interpreter.
    let src = "fn sum_to(n) {\n total = 0\n i = 0\n while i < n {\n total = total + i\n i = i + 1\n }\n return total\n}\nr = sum_to(100)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Int(4950));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("native (JIT) call 'sum_to'")), "should dispatch to native: {:?}", msgs);
}

#[cfg(feature = "jit")]
#[test]
fn jit_recursion_runs_native() {
    // v0.41: calls between JIT functions — a recursive fib runs entirely as machine code.
    let src = "fn fib(n) {\n if n < 2 { return n }\n else { return fib(n - 1) + fib(n - 2) }\n}\nr = fib(20)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Int(6765));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("JIT compiled 'fib'")), "fib should compile: {:?}", msgs);
    assert!(msgs.iter().any(|m| m.contains("native (JIT) call 'fib'")), "fib should dispatch native: {:?}", msgs);
}

#[cfg(feature = "jit")]
#[test]
fn jit_mutual_calls_run_native() {
    // Mutual recursion between two int-returning functions compiles as a batch (declared together).
    let src = "fn ev(n) {\n if n == 0 { return 1 }\n else { return od(n - 1) }\n}\nfn od(n) {\n if n == 0 { return 0 }\n else { return ev(n - 1) }\n}\na = ev(10)\nb = ev(7)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), Value::Int(1));
    assert_eq!(it.get("b").unwrap(), Value::Int(0));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("JIT compiled 'ev'")) && msgs.iter().any(|m| m.contains("JIT compiled 'od'")), "both should compile: {:?}", msgs);
}

#[cfg(feature = "jit")]
#[test]
fn jit_bool_return_falls_back() {
    // Parity: a bool-returning function must NOT be JIT'd (native i64 would turn `true` into 1).
    // This also guards the v0.37 hole fixed in v0.41 (`return a > b` was eligible and diverged).
    let src = "fn is_pos(n) { return n > 0 }\nr = is_pos(5)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Bool(true)); // tree-walker semantics preserved
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(!msgs.iter().any(|m| m.contains("JIT compiled 'is_pos'")), "bool return must be ineligible: {:?}", msgs);
}

#[cfg(feature = "jit")]
#[test]
fn jit_call_to_ineligible_fn_falls_back() {
    // The fixed point: f calls g; g uses strings (ineligible) → f is dropped too and both run interpreted.
    let src = "fn g(n) { return \"x\" + str(n) }\nfn f(n) { return g(n) }\nr = f(4)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), string("x4"));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(!msgs.iter().any(|m| m.contains("JIT compiled")), "neither should compile: {:?}", msgs);
}

#[cfg(feature = "jit")]
#[test]
fn jit_float_functions_run_native() {
    // v0.47: F64-ABI functions — float args dispatch natively; results match the tree-walker exactly.
    let src = "fn area(r) { return 3.5 * r * r }\na = area(2.0)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), Value::Float(14.0));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("JIT compiled 'area'") && m.contains("float fast path")), "area compiles as float: {:?}", msgs);
    assert!(msgs.iter().any(|m| m.contains("native (JIT) call 'area'")), "float args dispatch native: {:?}", msgs);
    // Int args fall back to the tree-walker (which promotes per-op) — same value, honest dispatch.
    assert_eq!(val("fn area(r) { return 3.5 * r * r }\na = area(2)", "a"), Value::Float(14.0));
}

#[cfg(feature = "jit")]
#[test]
fn jit_float_recursion_and_mixed_arith() {
    // Float recursion (same-ABI calls) + int literals promoted inside a float function.
    let src = "fn geo(x, n) {\n if n < 0.5 { return x }\n else { return geo(x * 0.5, n - 1.0) }\n}\nr = geo(64.0, 4.0)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Float(4.0));
    // mixed: `x * 2 + 0.5` — int literal promoted, matches the interpreter.
    let m = "fn mixed(x) { return x * 2 + 0.5 }\nr = mixed(1.25)";
    assert_eq!(val(m, "r"), Value::Float(3.0));
}

#[cfg(feature = "jit")]
#[test]
fn jit_float_division_stays_interpreted() {
    // Parity: native fdiv would give inf on /0.0 where the interpreter errors — so float division
    // is ineligible and such functions stay on the tree-walker.
    let src = "fn half(x) { return x / 2.0 }\nr = half(5.0)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Float(2.5));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(!msgs.iter().any(|m| m.contains("JIT compiled 'half'")), "float division must be ineligible: {:?}", msgs);
    assert!(eval_program("fn bad(x) { return x / 0.0 }\nr = bad(1.0)").is_err(), "division by 0.0 still a checked error");
}

#[cfg(feature = "jit")]
#[test]
fn jit_branches_and_comparisons() {
    // if/elif/else + comparisons compile to native and match.
    let src = "fn sign(n) {\n if n < 0 { return 0 - 1 } elif n > 0 { return 1 } else { return 0 }\n}\na = sign(0 - 5)\nb = sign(7)\nc = sign(0)";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), Value::Int(-1));
    assert_eq!(it.get("b").unwrap(), Value::Int(1));
    assert_eq!(it.get("c").unwrap(), Value::Int(0));
}

#[cfg(feature = "jit")]
#[test]
fn jit_falls_back_for_float_args() {
    // The function is eligible, but a float argument means the call falls back to the interpreter.
    let it = eval_program("fn f(x) { return x * 2 }\nr = f(2.5)").unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Float(5.0));
}

#[cfg(feature = "jit")]
#[test]
fn jit_ineligible_function_uses_interpreter() {
    // string concat → ineligible → never compiled; the interpreter handles it.
    let it = eval_program("fn g(s) { return s + \"!\" }\nr = g(\"hi\")").unwrap();
    assert_eq!(it.get("r").unwrap(), Value::Str("hi!".to_string()));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(!msgs.iter().any(|m| m.contains("JIT compiled 'g'")), "g must be ineligible: {:?}", msgs);
}

#[cfg(feature = "ai")]
#[test]
fn autodiff_simple_chain() {
    // y = (a * 3).sum();  dy/da = 3 (each element).  a=[1,2,3] → grad=[3,3,3], sum 9.
    let src = r#"
a = param([1, 2, 3])
y = (a * 3).sum()
y.backward()
gs = a.grad.sum().item()
"#;
    assert_eq!(val(src, "gs"), Value::Float(9.0));
}

#[test]
fn autodiff_only_params_track() {
    // Non-param tensors have no grad (not recorded in the graph).
    assert!(eval_program("a = tensor([1,2,3])\ny = (a*2).sum()\ny.backward()\ng = a.grad").is_err());
}

#[cfg(feature = "ai")]
#[test]
fn tensor_broadcasting_forward_and_backward() {
    // (2,3) + (1,3) bias broadcast
    assert_eq!(
        val("a=tensor([[1,2,3],[4,5,6]])\nb=tensor([[10,20,30]])\nx=(a+b).sum().item()", "x"),
        Value::Float(141.0)
    );
    // Broadcast backward: bias grad = sum over the batch axis → [2,2], sum 4
    let src = "x=tensor([[1,1],[1,1]])\nb=param([[1,2]])\ny=(x+b).sum()\ny.backward()\ng=b.grad.sum().item()";
    assert_eq!(val(src, "g"), Value::Float(4.0));
}

#[cfg(feature = "ai")]
#[test]
fn tensor_relu_and_transpose() {
    assert_eq!(val("x = relu(tensor([-2,-1,0,1,2])).sum().item()", "x"), Value::Float(3.0));
    assert_eq!(val("t = transpose(tensor([[1,2,3],[4,5,6]]))\ns = t.shape", "s"), arr(vec![int(3), int(2)]));
    // relu backward: grad passes only where x>0. a=[-1,2,-3,4] → grad [0,1,0,1], sum 2.
    let src = "a=param([-1, 2, -3, 4])\ny=relu(a).sum()\ny.backward()\ng=a.grad.sum().item()";
    assert_eq!(val(src, "g"), Value::Float(2.0));
}

#[cfg(feature = "ai")]
#[test]
fn mlp_two_layer_trains() {
    // A 2-layer MLP (relu) learns AND — loss goes nearly to 0.
    let src = r#"
x = tensor([[0,0],[0,1],[1,0],[1,1]])
target = tensor([[0],[0],[0],[1]])
W1 = param([[0.5,-0.3,0.2],[0.1,0.4,-0.2]])
b1 = param([[0,0,0]])
W2 = param([[0.4],[-0.3],[0.6]])
b2 = param([[0]])
fn forward(inp) { return matmul(relu(matmul(inp, W1) + b1), W2) + b2 }
final_loss = 999
for step in 0..400 {
    diff = forward(x) - target
    loss = (diff * diff).sum()
    loss.backward()
    grad_step(W1, 0.05)
    grad_step(b1, 0.05)
    grad_step(W2, 0.05)
    grad_step(b2, 0.05)
    final_loss = loss.item()
}
"#;
    let it = eval_program(src).unwrap();
    match it.get("final_loss").unwrap() {
        Value::Float(l) => assert!(l < 0.01, "MLP loss should drop enough: {}", l),
        _ => panic!("loss is not a float"),
    }
}

#[cfg(feature = "ai")]
#[test]
fn tensor_activations_forward_and_backward() {
    // forward (exact values)
    assert_eq!(val("x = sigmoid(tensor([0])).item()", "x"), Value::Float(0.5));
    assert_eq!(val("x = tanh(tensor([0])).item()", "x"), Value::Float(0.0));
    assert_eq!(val("x = exp(tensor([0])).item()", "x"), Value::Float(1.0));
    assert_eq!(val("x = log(tensor([1])).item()", "x"), Value::Float(0.0));
    // backward: σ'(0)=0.25, tanh'(0)=1, log'(1)=1
    assert_eq!(val("a=param([0])\ny=sigmoid(a).sum()\ny.backward()\ng=a.grad.item()", "g"), Value::Float(0.25));
    assert_eq!(val("a=param([0])\ny=tanh(a).sum()\ny.backward()\ng=a.grad.item()", "g"), Value::Float(1.0));
    assert_eq!(val("a=param([1])\ny=log(a).sum()\ny.backward()\ng=a.grad.item()", "g"), Value::Float(1.0));
}

#[cfg(feature = "ai")]
#[test]
fn logistic_classifier_trains_with_cross_entropy() {
    // sigmoid + log + BCE — logistic regression learns to classify AND.
    let src = r#"
x = tensor([[0,0],[0,1],[1,0],[1,1]])
y = tensor([[0],[0],[0],[1]])
w = param([[0],[0]])
b = param([[0]])
fn predict(inp) { return sigmoid(matmul(inp, w) + b) }
final_loss = 999
for step in 0..1500 {
    p = predict(x)
    loss = (y * log(p + 0.0001) + (1 - y) * log(1 - p + 0.0001)).mean() * -1
    loss.backward()
    grad_step(w, 0.5)
    grad_step(b, 0.5)
    final_loss = loss.item()
}
"#;
    let it = eval_program(src).unwrap();
    match it.get("final_loss").unwrap() {
        Value::Float(l) => assert!(l < 0.05, "BCE loss should drop: {}", l),
        _ => panic!("loss is not a float"),
    }
}

#[cfg(feature = "ai")]
#[test]
fn matmul_parallel_correctness() {
    // Large matrices use CPU multicore — the result must equal the single-threaded one (parallel correctness).
    // ones(64,64) @ ones(64,64): each element = 64, sum = 64³ = 262144.
    assert_eq!(val("r = matmul(ones([64,64]), ones([64,64])).sum().item()", "r"), Value::Float(262144.0));
    // Small ones (single-threaded) are exact too: [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]], sum 134.
    assert_eq!(val("r = matmul(tensor([[1,2],[3,4]]), tensor([[5,6],[7,8]])).sum().item()", "r"), Value::Float(134.0));
}

#[cfg(feature = "ai")]
#[test]
fn tensor_softmax_forward_and_backward() {
    // row sum = 1
    if let Value::Float(v) = val("p = softmax(tensor([1, 2, 3])).sum().item()", "p") {
        assert!((v - 1.0).abs() < 1e-4, "softmax row sum ≈ 1: {}", v);
    } else {
        panic!("not a float");
    }
    // backward: sum(softmax)=1 is constant → grad ≈ 0 (softmax Jacobian property)
    if let Value::Float(v) = val("a=param([1,2,3])\ny=softmax(a).sum()\ny.backward()\ng=a.grad.sum().item()", "g") {
        assert!(v.abs() < 1e-5, "grad of softmax sum ≈ 0: {}", v);
    } else {
        panic!("not a float");
    }
}

#[cfg(feature = "ai")]
#[test]
fn multiclass_softmax_classifier_trains() {
    let src = r#"
x = tensor([[2,0,0],[0,2,0],[0,0,2],[1,0,0],[0,1,0],[0,0,1]])
target = tensor([[1,0,0],[0,1,0],[0,0,1],[1,0,0],[0,1,0],[0,0,1]])
W = param([[0,0,0],[0,0,0],[0,0,0]])
b = param([[0,0,0]])
fn forward(inp) { return softmax(matmul(inp, W) + b) }
final_loss = 999
for step in 0..300 {
    p = forward(x)
    loss = (target * log(p + 0.0001)).sum() * -1
    loss.backward()
    grad_step(W, 0.3)
    grad_step(b, 0.3)
    final_loss = loss.item()
}
"#;
    let it = eval_program(src).unwrap();
    match it.get("final_loss").unwrap() {
        Value::Float(l) => assert!(l < 0.1, "multiclass loss should drop: {}", l),
        _ => panic!("loss is not a float"),
    }
}

#[cfg(feature = "ai")]
#[test]
fn tensor_device_residency() {
    assert_eq!(val("a = tensor([1,2,3])\nd = a.device", "d"), string("host"));
    assert_eq!(val("a = tensor([1,2,3])\nd = a.gpu().device", "d"), string("gpu"));
    assert_eq!(val("a = tensor([1,2,3])\nd = a.gpu().cpu().device", "d"), string("host"));
    // gpu tensor ops stay on gpu; host stays host.
    assert_eq!(val("a = tensor([1,2,3]).gpu()\nd = (a * 2).device", "d"), string("gpu"));
    assert_eq!(val("a = tensor([1,2,3])\nd = (a + 1).device", "d"), string("host"));
}

#[cfg(feature = "ai")]
#[test]
fn tensor_transfer_chain_eliminates_redundant() {
    // §4.3 core: when the chain stays on-device, only input H2D + output D2H — zero intermediate transfers.
    let src = r#"
x = tensor([[1,2],[3,4]]).gpu()
y = tensor([[1,0],[0,1]]).gpu()
h = matmul(x, y)
z = h * 2 + 1
out = z.cpu()
"#;
    let it = eval_program(src).unwrap();
    let msgs: Vec<&str> = it.channel.records.iter().map(|r| r.msg.as_str()).collect();
    // "D2H:" (with colon) = residency transfers (.cpu()/print). The real backend (`gpu` feature)
    // additionally illuminates each result's honest readback as "D2H result" — counted separately.
    let h2d = msgs.iter().filter(|m| m.contains("H2D:")).count();
    let d2h = msgs.iter().filter(|m| m.contains("D2H:")).count();
    let stayed = msgs.iter().filter(|m| m.contains("no transfer")).count();
    assert_eq!(h2d, 2, "only the 2 inputs should be H2D: {:?}", msgs);
    assert_eq!(d2h, 1, "only the 1 output should be a residency D2H: {:?}", msgs);
    assert!(stayed >= 1, "a 'no transfer' illumination is required during the chain: {:?}", msgs);
}

#[test]
fn match_expression() {
    assert_eq!(val(r#"x = match 1 { 0 => "a", 1 => "b", _ => "c" }"#, "x"), string("b"));
    assert_eq!(val(r#"x = match 9 { 0 => "a", _ => "c" }"#, "x"), string("c"));
    let src = r#"
enum Opt { Some(v) None }
fn unwrap_or(o, d) {
    return match o {
        Opt::Some(v) => v
        Opt::None => d
    }
}
a = unwrap_or(Opt::Some(42), 0)
b = unwrap_or(Opt::None, -1)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(42));
    assert_eq!(it.get("b").unwrap(), int(-1));
}

#[test]
fn error_values_and_predicates() {
    assert_eq!(val(r#"x = is_err(err("boom"))"#, "x"), boolean(true));
    assert_eq!(val("x = is_err(42)", "x"), boolean(false));
    assert_eq!(val(r#"x = err_msg(err("boom"))"#, "x"), string("boom"));
}

#[test]
fn error_propagation_with_question() {
    let src = r#"
fn safe_div(a, b) {
    if b == 0 { return err("div by zero") }
    return a / b
}
fn compute(a, b) {
    q = safe_div(a, b)?
    return q + 100
}
ok = compute(10, 2)
bad = compute(10, 0)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("ok").unwrap(), int(105));
    assert_eq!(it.get("bad").unwrap(), Value::Err(Box::new(string("div by zero"))));
}

#[test]
fn error_caught_without_question() {
    let src = r#"
fn may_fail(x) {
    if x < 0 { return err("negative") }
    return x * 2
}
fn handle(x) {
    r = may_fail(x)
    if is_err(r) { return -1 }
    return r
}
a = handle(5)
b = handle(-3)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(10));
    assert_eq!(it.get("b").unwrap(), int(-1));
}

#[test]
fn error_propagation_chains_through_callers() {
    let src = r#"
fn leaf(x) {
    if x == 0 { return err("zero!") }
    return x
}
fn mid(x) {
    v = leaf(x)?
    return v + 1
}
fn top(x) {
    v = mid(x)?
    return v + 10
}
good = top(5)
prop = top(0)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("good").unwrap(), int(16));
    assert_eq!(it.get("prop").unwrap(), Value::Err(Box::new(string("zero!"))));
}

#[test]
fn self_hosted_lexer() {
    // A lexer written in wide tokenizes wide (subset) source — a miniature of self-hosting.
    let src = r#"
fn is_digit(c) { return c >= "0" and c <= "9" }
fn is_alpha(c) { return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or c == "_" }
fn is_alnum(c) { return is_alpha(c) or is_digit(c) }
fn is_space(c) { return c == " " }

enum Token { Num(n) Name(s) Op(sym) End }

fn lex(src) {
    chars = src.chars()
    n = chars.len
    i = 0
    tokens = []
    while i < n {
        c = chars[i]
        if is_space(c) {
            i = i + 1
        } elif is_digit(c) {
            num = ""
            while i < n and is_digit(chars[i]) {
                num = num + chars[i]
                i = i + 1
            }
            tokens.push(Token::Num(int(num)))
        } elif is_alpha(c) {
            name = ""
            while i < n and is_alnum(chars[i]) {
                name = name + chars[i]
                i = i + 1
            }
            tokens.push(Token::Name(name))
        } else {
            tokens.push(Token::Op(c))
            i = i + 1
        }
    }
    tokens.push(Token::End)
    return tokens
}

fn show(tok) {
    match tok {
        Token::Num(n) => { return "Num(" + str(n) + ")" }
        Token::Name(s) => { return "Name(" + s + ")" }
        Token::Op(o) => { return "Op(" + o + ")" }
        Token::End => { return "End" }
    }
}

parts = []
for t in lex("a1 + 23 * x") {
    parts.push(show(t))
}
result = parts.join(" ")
count = parts.len
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(
        it.get("result").unwrap(),
        string("Name(a1) Op(+) Num(23) Op(*) Name(x) End")
    );
    assert_eq!(it.get("count").unwrap(), int(6));
}

#[test]
fn self_hosted_parser_calculator() {
    // Full lex→parse→eval pipeline written in wide (a real module chain). Verifies precedence, associativity, parentheses.
    let prog = wide::load_file(Path::new("tests/modules/calc_check.wide")).unwrap();
    let mut it = wide::Interp::new();
    it.run(&prog).unwrap();
    assert_eq!(it.get("r1").unwrap(), int(14)); // 2 + 3*4
    assert_eq!(it.get("r2").unwrap(), int(11)); // 2 + 3*(4-1)
    assert_eq!(it.get("r3").unwrap(), int(21)); // (1+2)*(3+4)
    assert_eq!(it.get("r4").unwrap(), int(74)); // 100 - 5*5 - 1  (left-associative)
    assert_eq!(it.get("tree").unwrap(), string("(1 + (2 * 3))")); // precedence tree
}

#[test]
fn self_hosted_lexer_modules_run() {
    // Whether the module-split self-hosted lexer runs to completion (import + enum + match combined).
    assert!(wide::load_file(Path::new("examples/selfhost/main.wide")).is_ok());
    let prog = wide::load_file(Path::new("examples/selfhost/main.wide")).unwrap();
    let mut it = wide::Interp::new();
    assert!(it.run(&prog).is_ok());
}

#[test]
fn typecheck_catches_errors_before_running() {
    let errs = |src: &str| wide::type_errors(src).unwrap();
    assert!(errs("print(y)").iter().any(|e| e.contains("undefined name 'y'")), "undefined name");
    assert!(errs("fn f(a) { return a }\nf(1, 2)").iter().any(|e| e.contains("arg")), "arity");
    assert!(errs("foo()").iter().any(|e| e.contains("undefined function")), "undefined function");
    assert!(errs("enum E { A }\ns = E::B").iter().any(|e| e.contains("variant")), "undefined variant");
    assert!(errs("break").iter().any(|e| e.contains("break")), "break outside loop");
    assert!(errs("return 5").iter().any(|e| e.contains("return")), "return outside function");
    // Collect several at once (don't stop at the first error).
    assert!(errs("print(a)\nprint(b)").len() >= 2, "collect multiple errors");
}

#[test]
fn std_module_gating() {
    let errs = |src: &str| wide::type_errors(src).unwrap();
    // Using it without import is a static error
    #[cfg(feature = "ai")]
    assert!(errs("a = tensor([1,2,3])").iter().any(|e| e.contains("std/ai")), "tensor gating");
    assert!(errs("h = heap()").iter().any(|e| e.contains("std/heap")), "heap gating");
    assert!(errs("s = set()").iter().any(|e| e.contains("std/set")), "set gating");
    // With import it passes
    #[cfg(feature = "ai")]
    assert!(errs("import \"std/ai\"\na = tensor([1,2,3])").is_empty(), "OK after ai import");
    assert!(errs("import \"std/heap\"\nh = heap()").is_empty(), "OK after heap import");
    // When the `ai` feature is compiled out, tensor builtins report the build-feature requirement.
    #[cfg(not(feature = "ai"))]
    assert!(errs("a = tensor([1,2,3])").iter().any(|e| e.contains("ai` build feature")), "ai build-feature gating");
    // Core needs no import
    assert!(errs("xs = [1,2,3]\nprint(len(xs))\nm = map{}").is_empty(), "core is always OK");
}

#[test]
fn set_operations() {
    // eval_program auto-enables std. Dedup, contains, len.
    let it = eval_program("s = set()\ns.add(1)\ns.add(1)\ns.add(2)\nhas = s.contains(1)\nno = s.contains(9)\nn = s.len").unwrap();
    assert_eq!(it.get("has").unwrap(), boolean(true));
    assert_eq!(it.get("no").unwrap(), boolean(false));
    assert_eq!(it.get("n").unwrap(), int(2));
}

#[test]
fn typecheck_no_false_positives() {
    // Every valid program must have zero errors (conservative check).
    let ok = |src: &str| {
        let e = wide::type_errors(src).unwrap();
        assert!(e.is_empty(), "false positive {:?} in:\n{}", e, src);
    };
    ok("x = 5\nprint(x)");
    ok("fn fib(n) { if n < 2 { return n }\nreturn fib(n - 1) + fib(n - 2) }\nprint(fib(10))");
    ok("xs = [1, 2, 3]\ntotal = 0\nfor i in xs { total = total + i }\nprint(total)");
    #[cfg(feature = "ai")]
    ok("import \"std/ai\"\na = tensor([1, 2, 3])\nprint(a.sum())"); // gating: import required
    ok("struct P { x, y }\np = P { x: 1, y: 2 }\nprint(p.x)");
    ok("enum Opt { Some(v) None }\nfn f(o) { return match o { Opt::Some(v) => v, Opt::None => 0 } }\nprint(f(Opt::Some(7)))");
    ok("for i in 0..3 { if i == 1 { continue }\nbreak }"); // nested loop control flow
}

// ---- v0.16: impl/methods + mutable strings (strbuf) ----

#[test]
fn impl_methods_basic() {
    // A method takes self to read fields, and also takes arguments.
    let src = r#"
struct Point { x, y }
impl Point {
    fn sum(self) { return self.x + self.y }
    fn scale(self, k) { return self.x * k }
}
p = Point { x: 3, y: 4 }
a = p.sum()
b = p.scale(10)
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(7));
    assert_eq!(it.get("b").unwrap(), int(30));
}

#[test]
fn impl_method_mutates_self_visible_to_caller() {
    // Reference semantics — self.field = ... is visible to the caller (accumulates across calls).
    let src = r#"
struct Counter { n }
impl Counter {
    fn inc(self) { self.n = self.n + 1 }
    fn add(self, k) {
        self.n = self.n + k
        return self.n
    }
}
c = Counter { n: 0 }
c.inc()
c.inc()
total = c.add(10)
final = c.n
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("total").unwrap(), int(12));
    assert_eq!(it.get("final").unwrap(), int(12));
}

#[test]
fn impl_method_arity_and_missing() {
    // Wrong arg count (excluding self) / missing method is a runtime error.
    assert!(eval_program("struct S { x }\nimpl S { fn f(self, a) { return a } }\ns = S { x: 1 }\nr = s.f()").is_err(), "arity");
    assert!(eval_program("struct S { x }\ns = S { x: 1 }\nr = s.nope()").is_err(), "missing method");
}

#[test]
fn impl_method_typecheck_resolves_body() {
    // Method bodies are name-resolved too (self and params are defined, undefined ones are caught). No false positives.
    let errs = wide::type_errors("struct S { x }\nimpl S { fn f(self, k) { return self.x + k } }").unwrap();
    assert!(errs.is_empty(), "false positive on valid method: {:?}", errs);
    let bad = wide::type_errors("struct S { x }\nimpl S { fn f(self) { return undefined_name } }").unwrap();
    assert!(bad.iter().any(|e| e.contains("undefined name")), "catch undefined name in method body: {:?}", bad);
}

#[test]
fn strbuf_builds_string() {
    // Mutable string builder — accumulate with push, extract with str(), .len.
    let src = r#"
b = strbuf()
for c in "hello" { b.push(c) }
b.push(" world")
s = b.str()
n = b.len
"#;
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("s").unwrap(), string("hello world"));
    assert_eq!(it.get("n").unwrap(), int(11));
}

#[test]
fn strbuf_clear_and_is_core() {
    // clear empties it. strbuf is core — no import needed (no gating).
    let it = eval_program("b = strbuf()\nb.push(\"abc\")\nb.clear()\nb.push(\"xy\")\ns = b.str()\nn = b.len").unwrap();
    assert_eq!(it.get("s").unwrap(), string("xy"));
    assert_eq!(it.get("n").unwrap(), int(2));
    assert!(wide::type_errors("b = strbuf()\nb.push(\"a\")").unwrap().is_empty(), "strbuf is core — no import needed");
}

// ---- v0.42: closures / first-class functions ----

#[test]
fn closures_basics() {
    // Lambda values, calling through a variable, and named functions as values.
    assert_eq!(val("add = fn(a, b) { return a + b }\nr = add(2, 3)", "r"), int(5));
    assert_eq!(val("fn double(x) { return x * 2 }\nd = double\nr = d(21)", "r"), int(42));
    // Higher-order: functions as arguments and as return values (closure factory).
    assert_eq!(val("fn double(x) { return x * 2 }\nfn apply(f, v) { return f(v) }\nr = apply(double, 7)", "r"), int(14));
    assert_eq!(val("fn make_adder(n) { return fn(x) { return x + n } }\nadd5 = make_adder(5)\nr = add5(100)", "r"), int(105));
}

#[test]
fn closures_capture_by_value_at_creation() {
    // Scalars are captured by value when the lambda is created — later reassignment is not seen.
    assert_eq!(val("k = 10\nf = fn(x) { return x + k }\nk = 99\nr = f(1)", "r"), int(11));
    // Collections are Rc (reference semantics), so a captured array *shares* mutations both ways.
    let src = "xs = [1, 2]\npush3 = fn() { xs.push(3) }\npush3()\nn = xs.len\ns = xs.sum()";
    assert_eq!(val(src, "n"), int(3));
    assert_eq!(val(src, "s"), int(6));
}

#[test]
fn closures_map_filter() {
    assert_eq!(val("xs = [1, 2, 3, 4]\nr = xs.map(fn(x) { return x * x }).sum()", "r"), int(30));
    assert_eq!(val("xs = [1, 2, 3, 4]\nr = xs.filter(fn(x) { return x > 2 }).len", "r"), int(2));
    // map does not mutate the source array.
    assert_eq!(val("xs = [1, 2]\nys = xs.map(fn(x) { return x + 1 })\nr = xs.sum()", "r"), int(3));
    assert!(eval_program("xs = [1]\nr = xs.filter(fn(x) { return x + 1 })").is_err(), "filter fn must return bool");
    assert!(eval_program("xs = [1]\nr = xs.map(5)").is_err(), "map takes a function value");
}

#[test]
fn closures_errors_and_checks() {
    assert!(eval_program("f = 5\nr = f(1)").is_err(), "calling a non-fn value errors");
    assert!(eval_program("f = fn(a, b) { return a + b }\nr = f(1)").is_err(), "closure arity checked at call");
    // Static checker: fn-as-value and variable calls pass; undefined names still caught.
    assert!(wide::type_errors("fn d(x) { return x }\nf = d\nr = f(3)").unwrap().is_empty(), "fn-as-value passes the checker");
    assert!(wide::type_errors("f = fn(x) { return x + 1 }\nr = f(2)").unwrap().is_empty(), "lambda passes the checker");
    assert!(wide::type_errors("f = fn(x) { return x + nope }").unwrap().iter().any(|e| e.contains("undefined name")), "lambda bodies are checked");
    // `?` propagates out of a closure like out of a named function.
    let src = "f = fn(x) { if x == 0 { return err(\"zero\") }\nreturn 10 / x }\nfn use(x) { q = f(x)?\nreturn q + 1 }\nr = use(0)\ne = is_err(r)\nok = use(5)";
    assert_eq!(val(src, "e"), boolean(true));
    assert_eq!(val(src, "ok"), int(3));
}

#[test]
fn closures_vm_rejects_clearly() {
    let e = match wide::eval_program_vm("f = fn(x) { return x }") {
        Err(e) => e,
        Ok(_) => panic!("VM must reject closures"),
    };
    assert!(e.contains("not supported by the VM yet") || e.contains("closures"), "VM must reject closures clearly: {}", e);
}

// ---- v0.43: file I/O (std/fs) ----

#[test]
fn fs_roundtrip_lines_append_remove() {
    let dir = std::env::temp_dir().join("wide_fs_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("t1.txt").to_string_lossy().replace('\\', "/");
    // "a\nb" → 3 bytes, 2 lines; append "!" → 4 bytes. remove → gone; reading it back is an error-value.
    let src = format!(
        "import \"std/fs\"\nwrite_file(\"{p}\", \"a\\nb\")\nn = read_file(\"{p}\").len\nm = read_lines(\"{p}\").len\nappend_file(\"{p}\", \"!\")\nk = read_file(\"{p}\").len\ne1 = file_exists(\"{p}\")\nremove_file(\"{p}\")\ne2 = file_exists(\"{p}\")\nbe = is_err(read_file(\"{p}\"))"
    );
    let it = eval_program(&src).unwrap();
    assert_eq!(it.get("n").unwrap(), int(3));
    assert_eq!(it.get("m").unwrap(), int(2));
    assert_eq!(it.get("k").unwrap(), int(4));
    assert_eq!(it.get("e1").unwrap(), boolean(true));
    assert_eq!(it.get("e2").unwrap(), boolean(false));
    assert_eq!(it.get("be").unwrap(), boolean(true));
    // I/O cost is illuminated (byte counts).
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("fs write") && m.contains("3 B")), "write illuminated: {:?}", msgs);
    assert!(msgs.iter().any(|m| m.contains("fs read")), "read illuminated: {:?}", msgs);
}

#[test]
fn fs_error_values_propagate_with_question() {
    // A missing file is an error-*value* — `?` propagates it like any other error union.
    let src = "import \"std/fs\"\nfn load(p) { s = read_file(p)?\nreturn s.len }\nr = load(\"definitely_missing_file.xyz\")\ne = is_err(r)";
    assert_eq!(val(src, "e"), boolean(true));
}

#[test]
fn fs_requires_import_gating() {
    let errs = wide::type_errors("s = read_file(\"x.txt\")").unwrap();
    assert!(errs.iter().any(|e| e.contains("std/fs")), "fs builtins are gated: {:?}", errs);
    assert!(wide::type_errors("import \"std/fs\"\ns = file_exists(\"x.txt\")").unwrap().is_empty());
}

#[test]
fn fs_works_on_the_vm_too() {
    // fs lives in the shared runtime::value_builtin → the bytecode VM gets it for free (parity).
    let dir = std::env::temp_dir().join("wide_fs_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("t2.txt").to_string_lossy().replace('\\', "/");
    let src = format!("import \"std/fs\"\nwrite_file(\"{p}\", \"vm\")\nn = read_file(\"{p}\").len\nremove_file(\"{p}\")");
    let vm = wide::eval_program_vm(&src).unwrap();
    assert_eq!(vm.get("n").unwrap(), int(2));
}

// ---- v0.44: negative indexes + slices ----

#[test]
fn negative_indexes() {
    assert_eq!(val("xs = [10, 20, 30]\nr = xs[-1]", "r"), int(30));
    assert_eq!(val("xs = [10, 20, 30]\nr = xs[-3]", "r"), int(10));
    assert_eq!(val("s = \"abc\"\nr = s[-1]", "r"), string("c"));
    // negative write too
    assert_eq!(val("xs = [1, 2, 3]\nxs[-1] = 99\nr = xs[2]", "r"), int(99));
    // still strictly bounds-checked
    assert!(eval_program("xs = [1, 2]\nr = xs[-3]").is_err(), "too-negative index blocked");
}

#[test]
fn slices_arrays_and_strings() {
    assert_eq!(val("xs = [1, 2, 3, 4, 5]\nr = xs[1..3].sum()", "r"), int(5));
    assert_eq!(val("xs = [1, 2, 3, 4, 5]\nr = xs[0..0].len", "r"), int(0));
    assert_eq!(val("s = \"hello\"\nr = s[1..4]", "r"), string("ell"));
    // negative endpoints count from the end
    assert_eq!(val("xs = [1, 2, 3, 4, 5]\nr = xs[-2..5].sum()", "r"), int(9));
    assert_eq!(val("s = \"hello\"\nr = s[0..-1]", "r"), string("hell"));
    // a slice is a *copy* — mutating it leaves the source untouched
    assert_eq!(val("xs = [1, 2, 3]\nys = xs[0..2]\nys[0] = 99\nr = xs[0]", "r"), int(1));
    // strict bounds (no silent clamping), and slices are not assignable
    assert!(eval_program("xs = [1, 2]\nr = xs[0..5]").is_err(), "slice out of range blocked");
    assert!(eval_program("xs = [1, 2]\nxs[0..1] = [9]").is_err(), "slice assignment rejected");
}

#[test]
fn vm_negative_indexes_and_slices_match() {
    // Indexing lives in the shared runtime — the VM gets the same semantics automatically.
    let a = "xs = [10, 20, 30]\nr = xs[-1]";
    assert_eq!(vm_val(a, "r"), val(a, "r"));
    let b = "xs = [1, 2, 3, 4, 5]\nr = xs[1..3].sum()";
    assert_eq!(vm_val(b, "r"), val(b, "r"));
    let c = "s = \"hello\"\nr = s[1..-1]";
    assert_eq!(vm_val(c, "r"), val(c, "r"));
}

// ---- v0.45: newlines inside ( ) / [ ] (multi-line literals & argument lists) ----

#[test]
fn multiline_literals_and_args() {
    assert_eq!(val("xs = [1, 2,\n 3,\n 4]\nr = xs.sum()", "r"), int(10));
    assert_eq!(val("fn add(a, b) { return a + b }\nr = add(\n 1,\n 2\n)", "r"), int(3));
    // Blocks inside parens keep their newlines (a multi-line lambda body as a call argument).
    assert_eq!(val("xs = [1, 2]\nys = xs.map(fn(x) {\n y = x * 2\n return y\n})\nr = ys.sum()", "r"), int(6));
    // Nested: brackets inside parens inside brackets.
    assert_eq!(val("xs = [[1,\n 2], [3,\n 4]]\nr = xs[1][0]", "r"), int(3));
    // VM parity (lexing is shared, but verify end to end).
    assert_eq!(vm_val("xs = [1,\n 2,\n 3]\nr = xs.sum()", "r"), int(6));
}

// ---- v0.48: explicit borrows (&x / &mut x), scope lifetime, @trust ----

#[test]
fn explicit_shared_borrow_blocks_mutation_until_scope_end() {
    // While `r = &xs` lives (its scope), mutating xs conflicts; after the scope it's fine.
    assert!(eval_program("xs = [1]\nr = &xs\nxs.push(2)").is_err(), "mutation while shared-borrowed blocked");
    let src = "xs = [1]\nif true {\n r = &xs\n s = r[0]\n}\nxs.push(2)\nn = xs.len";
    assert_eq!(val(src, "n"), int(2)); // claim released at scope end
}

#[test]
fn exclusive_borrow_xor_rules() {
    // &mut XOR &: no second borrow, no iteration, while an exclusive claim lives.
    assert!(eval_program("xs = [1]\nm = &mut xs\nr = &xs").is_err(), "& while &mut blocked");
    assert!(eval_program("xs = [1]\nm = &mut xs\nn = &mut xs").is_err(), "&mut while &mut blocked");
    assert!(eval_program("xs = [1, 2]\nm = &mut xs\nfor x in xs { print(x) }").is_err(), "iteration while &mut blocked");
    assert!(eval_program("xs = [1]\nr = &xs\nm = &mut xs").is_err(), "&mut while shared blocked");
    // Mutation through the exclusive borrow is fine (attribution is the static tier's job — honest).
    assert_eq!(val("xs = [1]\nm = &mut xs\nm.push(2)\nn = xs.len", "n"), int(2));
}

#[test]
fn borrows_release_at_function_return() {
    // A borrow made inside a function frame dies when the frame returns — whatever the exit path.
    let src = "xs = [1]\nfn peek(a) {\n r = &a\n return r[0]\n}\nv = peek(xs)\nxs.push(2)\nn = xs.len";
    assert_eq!(val(src, "n"), int(2));
}

#[test]
fn trust_turns_checks_off_with_warn() {
    // @trust: the conflict is not blocked, and the trust is illuminated as WARN.
    let src = "xs = [1]\nr = &xs\n@trust xs.push(2)\nn = xs.len";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("n").unwrap(), int(2));
    let warns: Vec<String> = it.channel.records.iter().filter(|r| matches!(r.level, wide::lumen::Level::Warn)).map(|r| r.msg.clone()).collect();
    assert!(warns.iter().any(|m| m.contains("@trust")), "trust must be illuminated: {:?}", warns);
}

#[test]
fn show_provenance_reflects_exclusive_access() {
    let src = "xs = [1]\nm = &mut xs\n@show provenance xs";
    let it = eval_program(src).unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("access exclusive (&mut)")), "provenance shows exclusive: {:?}", msgs);
}

#[test]
fn explicit_borrows_rejected_by_vm_clearly() {
    let e = match wide::eval_program_vm("xs = [1]\nm = &mut xs") {
        Err(e) => e,
        Ok(_) => panic!("VM must reject &mut"),
    };
    assert!(e.contains("not supported by the VM yet"), "clear VM rejection: {}", e);
}

// ---- v0.49: borrow static-proof tier (the full gradient: proof → guard → trust) ----

#[test]
fn static_tier_catches_straight_line_conflicts() {
    // A conflict the runtime guard would certainly hit, in straight-line code → caught before run.
    let errs = wide::type_errors("xs = [1]\nr = &xs\nxs.push(2)").unwrap();
    assert!(errs.iter().any(|e| e.contains("caught before run")), "definite conflict is a compile error: {:?}", errs);
    let errs2 = wide::type_errors("xs = [1]\nm = &mut xs\nr = &xs").unwrap();
    assert!(errs2.iter().any(|e| e.contains("caught before run")), "& while &mut caught statically: {:?}", errs2);
    let errs3 = wide::type_errors("xs = [1]\nm = &mut xs\nfor x in xs { print(x) }").unwrap();
    assert!(errs3.iter().any(|e| e.contains("caught before run")), "iteration while &mut caught statically: {:?}", errs3);
}

#[test]
fn static_tier_no_false_positives() {
    // A conflict inside a conditional may never execute — NOT a compile error (principle 6);
    // the runtime guard still blocks it when it does execute.
    assert!(wide::type_errors("xs = [1]\nr = &xs\nif false { xs.push(2) }").unwrap().is_empty());
    assert!(eval_program("xs = [1]\nr = &xs\nif true { xs.push(2) }").is_err(), "runtime guard still fires");
    // Dead code after return is not flagged.
    assert!(wide::type_errors("fn f(xs) { r = &xs\nreturn r[0]\nxs.push(1) }\nv = f([1])").unwrap().is_empty());
    // @trust exempts its statement from the static conflict too.
    assert!(wide::type_errors("xs = [1]\nr = &xs\n@trust xs.push(2)").unwrap().is_empty());
}

#[test]
fn proven_borrows_skip_the_guard_with_cost_zero() {
    // Only reads in the region + a globally quiet name → proven: no runtime guard, illuminated.
    let src = "xs = [1, 2]\nif true {\n r = &xs\n a = r[0] + xs.len\n}\nxs.push(3)\nn = xs.len";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("n").unwrap(), int(3));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("statically proven safe — cost 0")), "proof illuminated: {:?}", msgs);
    assert!(!msgs.iter().any(|m| m.contains("shared borrow of xs (&) (runtime guard")), "no guard for proven claim: {:?}", msgs);
}

#[test]
fn unprovable_borrows_stay_on_the_guard_tier() {
    // The name escapes into a call → can't prove → runtime guard (v0.48 behavior preserved).
    let src = "fn touch(a) { return a.len }\nys = [1]\ns = &ys\nk = touch(ys)";
    let it = eval_program(src).unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("runtime guard")), "escape demotes to guard: {:?}", msgs);
    // Soundness: an outer &mut means a later &-claim must NOT be proven-skipped (the guard must fire).
    assert!(eval_program("xs = [1]\nm = &mut xs\nif true { r = &xs }").is_err(), "nested & vs outer &mut still blocked");
}

#[test]
fn show_provenance_keeps_claims_observable() {
    // @show provenance wants the live table — claims on shown names stay guard-tier (not proven away).
    let src = "xs = [1]\nm = &mut xs\n@show provenance xs";
    let it = eval_program(src).unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("access exclusive (&mut)")), "provenance shows the claim: {:?}", msgs);
}

// ---- v0.50: real GPU compute backend (wgpu, `gpu` feature) ----

#[cfg(feature = "gpu")]
#[test]
fn gpu_matmul_matches_cpu() {
    if wide::gpu::ctx().is_none() {
        eprintln!("no gpu adapter available — skipping (honest: nothing was tested)");
        return;
    }
    // The same matmul on gpu-resident tensors must produce exactly the CPU result.
    let cpu = "a = tensor([[1, 2, 3], [4, 5, 6]])\nw = tensor([[1, 0], [0, 1], [1, 1]])\nx = matmul(a, w).sum().item()";
    let gpu = "a = tensor([[1, 2, 3], [4, 5, 6]]).gpu()\nw = tensor([[1, 0], [0, 1], [1, 1]]).gpu()\nx = matmul(a, w).sum().item()";
    assert_eq!(val(gpu, "x"), val(cpu, "x"));
    assert_eq!(val(gpu, "x"), Value::Float(30.0));
    // Illumination shows the real backend.
    let it = eval_program(gpu).unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("gpu (wgpu:")), "real gpu backend illuminated: {:?}", msgs);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_chain_reuses_resident_buffers() {
    if wide::gpu::ctx().is_none() {
        eprintln!("no gpu adapter available — skipping");
        return;
    }
    // A chained matmul reuses the previous result's device buffer — no re-upload (§4.3, now real).
    let src = "a = ones([4, 4]).gpu()\nb = ones([4, 4]).gpu()\nc = matmul(matmul(a, b), b)\ns = c.sum().item()";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("s").unwrap(), Value::Float(256.0)); // ones: (4·1)=4 per elem, then 4·4·4=16 per elem, ×16 elems
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("resident on device — no transfer")), "chain reuses buffers: {:?}", msgs);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_elementwise_matches_cpu_and_stays_resident() {
    if wide::gpu::ctx().is_none() {
        eprintln!("no gpu adapter available — skipping");
        return;
    }
    // v0.51: +,-,*,/ (tensor∘tensor same-shape and tensor∘scalar, both operand orders) on the GPU.
    let cpu = "a = tensor([[1, 2], [3, 4]])\nb = tensor([[10, 20], [30, 40]])\nx = ((a + b) * a - b).sum().item()\ny = ((10 - a) * 2 + (a / 2)).sum().item()";
    let gpu = "a = tensor([[1, 2], [3, 4]]).gpu()\nb = tensor([[10, 20], [30, 40]]).gpu()\nx = ((a + b) * a - b).sum().item()\ny = ((10 - a) * 2 + (a / 2)).sum().item()";
    assert_eq!(val(gpu, "x"), val(cpu, "x"));
    assert_eq!(val(gpu, "y"), val(cpu, "y"));
    // The whole chain stays resident — elementwise results carry device buffers (no re-upload).
    let it = eval_program("a = ones([4, 4]).gpu()\nc = matmul(a, a) * 2 + 1\ns = c.sum().item()").unwrap();
    assert_eq!(it.get("s").unwrap(), Value::Float(144.0)); // (4·2+1)=9 × 16 elems
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("elementwise") && m.contains("wgpu")), "elementwise on gpu: {:?}", msgs);
    let uploads = msgs.iter().filter(|m| m.contains("H2D")).count();
    assert_eq!(uploads, 1, "only the initial upload — chain re-uploads nothing: {:?}", msgs);
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_training_still_converges() {
    if wide::gpu::ctx().is_none() {
        eprintln!("no gpu adapter available — skipping");
        return;
    }
    // Forward on GPU (x resident; w uploads lazily and re-caches after each grad_step invalidation),
    // backward on CPU (autodiff reads the eagerly-downloaded host data). Must still converge.
    // (Note: `w.gpu()` would train a *copy* — .gpu() clones, so grads would land on the clone.)
    let src = "x = tensor([[1, 2], [3, 4]]).gpu()\nw = param([[0], [0]])\ntarget = tensor([[5], [11]])\ni = 0\nwhile i < 60 {\n d = matmul(x, w) - target\n l = (d * d).mean()\n l.backward()\n grad_step(w, 0.04)\n i = i + 1\n}\nxc = x.cpu()\nfinal = ((matmul(xc, w) - target) * (matmul(xc, w) - target)).mean().item()";
    let it = eval_program(src).unwrap();
    let f = match it.get("final").unwrap() {
        Value::Float(x) => x,
        other => panic!("loss should be float: {:?}", other),
    };
    assert!(f < 0.5, "training over the gpu backend should converge (final {})", f);
}

// ---- v0.53: std/ml — built-in models (scikit-learn style) + EDA helpers ----

#[cfg(feature = "ai")]
#[test]
fn ml_logistic_regression_fits_and_scores() {
    // OR gate: fit converges; predictions land on the right side of 0.5; score = BCE.
    let src = "import \"std/ml\"\nx = tensor([[0, 0], [0, 1], [1, 0], [1, 1]])\ny = tensor([[0], [1], [1], [1]])\nm = logistic_regression()\nloss = m.fit(x, y, 300, 0.1)\ns = m.score(x, y)\np0 = m.predict(tensor([[0, 0]])).item()\np1 = m.predict(tensor([[1, 1]])).item()";
    let it = eval_program(src).unwrap();
    let f = |k: &str| match it.get(k).unwrap() { Value::Float(x) => x, v => panic!("{k} not float: {v:?}") };
    assert!(f("loss") < 0.1, "BCE should converge (got {})", f("loss"));
    assert!(f("s") < 0.1, "score is BCE (got {})", f("s"));
    assert!(f("p0") < 0.5 && f("p1") > 0.5, "predictions separate the classes ({}, {})", f("p0"), f("p1"));
}

#[cfg(feature = "ai")]
#[test]
fn ml_linear_regression_fits() {
    // y = 1 + 2a + 3b — the model recovers it well enough to extrapolate.
    let src = "import \"std/ml\"\nx = tensor([[1, 1], [2, 1], [3, 2], [4, 3], [5, 5]])\ny = tensor([[6], [8], [13], [18], [26]])\nm = linear_regression()\nloss = m.fit(x, y, 500, 0.1)\npred = m.predict(tensor([[6, 4]])).item()";
    let it = eval_program(src).unwrap();
    let f = |k: &str| match it.get(k).unwrap() { Value::Float(x) => x, v => panic!("{k} not float: {v:?}") };
    assert!(f("loss") < 0.1, "MSE should converge (got {})", f("loss"));
    assert!((f("pred") - 25.0).abs() < 1.0, "predict(6,4) ~ 25 (got {})", f("pred"));
}

#[cfg(feature = "ai")]
#[test]
fn ml_read_csv_and_describe() {
    let dir = std::env::temp_dir().join("wide_ml_test");
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("d.csv").to_string_lossy().replace('\\', "/");
    std::fs::write(dir.join("d.csv"), "a,b\n1,10\n2,20\n3,30\n4,40\n").unwrap();
    let src = format!("import \"std/ml\"\nt = read_csv(\"{p}\")\nsh = t.shape\ns = t.sum().item()\ndescribe(t)");
    let it = eval_program(&src).unwrap();
    assert_eq!(it.get("sh").unwrap(), arr(vec![int(4), int(2)])); // header skipped
    assert_eq!(it.get("s").unwrap(), Value::Float(110.0));
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("csv read") && m.contains("header skipped")), "csv illuminated: {:?}", msgs);
    // failure is an error-value
    assert_eq!(val("import \"std/ml\"\ne = is_err(read_csv(\"no_such.csv\"))", "e"), boolean(true));
    let _ = std::fs::remove_file(dir.join("d.csv"));
}

#[cfg(feature = "ai")]
#[test]
fn ml_sqrt_tensor_is_differentiable() {
    assert_eq!(val("t = tensor([4, 9, 16])\ns = sqrt(t).sum().item()", "s"), Value::Float(9.0));
    // d/dx sqrt = 0.5/sqrt(x): at 4 → 0.25, at 16 → 0.125; sum = 0.375
    let src = "w = param([4, 16])\nl = sqrt(w).sum()\nl.backward()\ng = w.grad.sum().item()";
    assert_eq!(val(src, "g"), Value::Float(0.375));
}

#[test]
fn ml_requires_import() {
    let errs = wide::type_errors("m = logistic_regression()").unwrap();
    assert!(errs.iter().any(|e| e.contains("undefined function")), "no defs without the import: {:?}", errs);
    let errs2 = wide::type_errors("t = read_csv(\"x.csv\")").unwrap();
    assert!(errs2.iter().any(|e| e.contains("std/ml")), "read_csv is gated: {:?}", errs2);
}

// ---- v0.53: illumination memory — identical records aggregate instead of accumulating ----

#[test]
fn illumination_aggregates_in_loops() {
    // 10k identical allocations must produce ONE record with a count, not 10k records
    // (unbounded accumulation made long-running programs balloon — measured 55 MB at 300k iterations).
    let src = "i = 0\nwhile i < 10000 {\n xs = [1, 2, 3]\n i = i + 1\n}\nn = i";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("n").unwrap(), int(10000));
    assert!(it.channel.records.len() < 10, "records must aggregate: {} stored", it.channel.records.len());
    assert!(it.channel.records.iter().any(|r| r.count == 10000), "the repeated record carries its count");
    assert_eq!(it.channel.truncated, 0);
}

// ---- v0.54: class declarations (struct + impl in one, with associated functions) ----

#[test]
fn class_constructor_methods_fields() {
    let src = "class Counter {\n n, step\n fn new(start) {\n return Counter { n: start, step: 1 }\n }\n fn tick(self) {\n self.n = self.n + self.step\n return self.n\n }\n}\nc = Counter::new(10)\na = c.tick()\nb = c.tick()\nn = c.n";
    let it = eval_program(src).unwrap();
    assert_eq!(it.get("a").unwrap(), int(11));
    assert_eq!(it.get("b").unwrap(), int(12));
    assert_eq!(it.get("n").unwrap(), int(12));
    // A class still works as a plain struct literal too (it *is* a struct + impl).
    assert_eq!(val("class P { x, y\n fn s(self) { return self.x + self.y }\n}\np = P { x: 3, y: 4 }\nr = p.s()", "r"), int(7));
}

#[test]
fn class_vm_parity_no_infinite_loop() {
    // Regression: the VM's associated-function dispatch `continue`d without advancing the instruction
    // pointer — the op re-executed forever and the stack grew until memory ran out (user-reported).
    let src = "class Counter {\n n, step\n fn new(start) {\n return Counter { n: start, step: 1 }\n }\n fn tick(self) {\n self.n = self.n + self.step\n return self.n\n }\n}\nc = Counter::new(10)\na = c.tick()\nb = c.tick()\nn = c.n";
    assert_eq!(vm_val(src, "a"), int(11));
    assert_eq!(vm_val(src, "b"), int(12));
    assert_eq!(vm_val(src, "n"), val(src, "n"));
}

#[test]
fn class_errors() {
    // Fields must precede methods; enum construction is unaffected by associated-fn dispatch.
    assert!(wide::parse("class C {\n fn m(self) { return 1 }\n x\n}").is_err(), "fields after methods rejected");
    assert_eq!(val("enum E { A(v) }\nx = match E::A(7) { E::A(v) => v, _ => 0 }", "x"), int(7));
}

// ---- v0.18: I/O (cout / cin) ----

#[test]
fn cin_reads_and_auto_types() {
    // cin splits on whitespace and auto-types each token: int → float → str.
    let it = wide::eval_program_with_input("cin >> a >> b >> c", "10 3.5 hello").unwrap();
    assert_eq!(it.get("a").unwrap(), int(10));
    assert_eq!(it.get("b").unwrap(), Value::Float(3.5));
    assert_eq!(it.get("c").unwrap(), string("hello"));
}

#[test]
fn cin_into_index_then_compute() {
    // cin can target an index lvalue; tokens may span multiple lines.
    let it = wide::eval_program_with_input("xs = [0, 0]\ncin >> xs[0] >> xs[1]\ns = xs[0] + xs[1]", "4\n5").unwrap();
    assert_eq!(it.get("s").unwrap(), int(9));
}

#[test]
fn cin_runs_out_of_input_errors() {
    assert!(wide::eval_program_with_input("cin >> a >> b", "1").is_err(), "too few input tokens errors");
}

#[test]
fn io_typecheck() {
    // cin defines its targets (no false positive); cout checks its expressions.
    assert!(wide::type_errors("cin >> a\nb = a + 1").unwrap().is_empty(), "cin defines a");
    assert!(wide::type_errors("cout << undefined_x").unwrap().iter().any(|e| e.contains("undefined name")), "cout checks exprs");
}

// ---- v0.19: bytecode compiler + VM (stage 2) — parity with the tree-walker ----

fn vm_val(src: &str, var: &str) -> Value {
    wide::eval_program_vm(src)
        .unwrap_or_else(|e| panic!("vm failed: {}\n--- source ---\n{}", e, src))
        .get(var)
        .unwrap_or_else(|| panic!("variable '{}' not found", var))
}

#[test]
fn vm_core_matches_tree_walker() {
    // Same programs, both backends → identical results.
    let cases = [
        ("x = 2 + 3 * 4", "x"),
        ("x = (2 + 3) * 4 - 1", "x"),
        ("x = 7 / 2", "x"),
        ("x = 1 / 2.0", "x"),
        ("x = 10 - 2 - 3", "x"),
        ("x = -5 + 2", "x"),
        ("x = 3 < 5", "x"),
        ("x = 3 == 3", "x"),
        ("x = true and not false", "x"),
        ("x = false or true", "x"),
        (r#"x = "wi" + "de""#, "x"),
    ];
    for (src, var) in cases {
        assert_eq!(vm_val(src, var), val(src, var), "backend mismatch in: {}", src);
    }
}

#[test]
fn vm_functions_and_control_flow() {
    let fib = "fn fib(n) { if n < 2 { return n }\nreturn fib(n - 1) + fib(n - 2) }\nx = fib(12)";
    assert_eq!(vm_val(fib, "x"), int(144));
    assert_eq!(vm_val(fib, "x"), val(fib, "x"));

    let sum = "t = 0\nfor i in 1..101 { t = t + i }";
    assert_eq!(vm_val(sum, "t"), int(5050));

    // while + continue + break — compare both backends.
    let w = "n = 0\ni = 0\nwhile i < 10 { i = i + 1\nif i == 5 { continue }\nif i == 8 { break }\nn = n + i }";
    assert_eq!(vm_val(w, "n"), val(w, "n"));

    // mutual recursion.
    let mr = "fn ev(n) { if n == 0 { return true }\nreturn od(n - 1) }\nfn od(n) { if n == 0 { return false }\nreturn ev(n - 1) }\nx = ev(10)";
    assert_eq!(vm_val(mr, "x"), boolean(true));
}

#[test]
fn vm_runtime_errors_match() {
    assert!(wide::eval_program_vm("x = 1 / 0").is_err(), "division by zero");
    assert!(wide::eval_program_vm("x = y").is_err(), "undefined name");
    assert!(wide::eval_program_vm("x = 1 + true").is_err(), "type mismatch");
}

#[test]
fn vm_arrays_indexing_and_iteration() {
    // Arrays, index get/set, for-in (array + string), methods, .len — all parity with the tree-walker (v0.20).
    let src = "xs = [3, 1, 4, 1, 5]\nxs.push(9)\nxs[0] = 30\nt = 0\nfor x in xs { t = t + x }\nn = xs.len";
    assert_eq!(vm_val(src, "t"), int(50));
    assert_eq!(vm_val(src, "n"), int(6));
    assert_eq!(vm_val(src, "t"), val(src, "t"));
    let joined = "xs = [3, 1, 2]\nxs.sort()\ns = xs.join(\"-\")";
    assert_eq!(vm_val(joined, "s"), string("1-2-3"));
    let str_iter = "cnt = 0\nfor c in \"wide\" { cnt = cnt + 1 }\nu = \"ab\".upper()";
    assert_eq!(vm_val(str_iter, "cnt"), int(4));
    assert_eq!(vm_val(str_iter, "u"), string("AB"));
}

#[test]
fn vm_maps() {
    // Maps: literal, index get/set, get/contains/keys, .len — parity with the tree-walker (v0.21).
    let src = "m = map{}\nm[\"a\"] = 1\nm[\"b\"] = 2\nm[\"a\"] = m[\"a\"] + 10\nx = m[\"a\"]\nd = m.get(\"z\", -1)\nn = m.len";
    assert_eq!(vm_val(src, "x"), int(11));
    assert_eq!(vm_val(src, "d"), int(-1));
    assert_eq!(vm_val(src, "n"), int(2));
    assert_eq!(vm_val(src, "x"), val(src, "x"));
}

#[test]
fn vm_builtins_and_errors() {
    // Core builtins via the shared `runtime` module + `?` error propagation — parity with the tree-walker (v0.22).
    assert_eq!(vm_val("x = len(\"hello\") + abs(-3)", "x"), int(8));
    assert_eq!(vm_val("x = max(3, 9, 5)", "x"), int(9));
    assert_eq!(vm_val("x = int(\"42\") + pow(2, 5)", "x"), int(74));
    assert_eq!(vm_val("x = hex(255)", "x"), string("0xff"));
    // ? error propagation
    let src = "fn sd(a, b) { if b == 0 { return err(\"z\") }\nreturn a / b }\nfn c(a, b) { q = sd(a, b)?\nreturn q + 1 }\nr = c(10, 0)\ne = is_err(r)\nm = err_msg(r)\nok = c(10, 2)";
    assert_eq!(vm_val(src, "e"), boolean(true));
    assert_eq!(vm_val(src, "m"), string("z"));
    assert_eq!(vm_val(src, "ok"), int(6));
    assert_eq!(vm_val(src, "ok"), val(src, "ok"));
    // eq parity fix: incomparable types error (matches tree-walker), not silently false.
    assert!(wide::eval_program_vm("x = 3 == \"a\"").is_err(), "incomparable == errors");
    assert!(eval_program("x = 3 == \"a\"").is_err(), "tree-walker too");
}

#[test]
fn vm_struct_enum_match_impl() {
    // struct + impl methods + field get/set (v0.23) — parity.
    let s = "struct P { x, y }\nimpl P { fn s(self) { return self.x + self.y } }\np = P { x: 3, y: 4 }\np.x = 10\na = p.s()\nb = p.x";
    assert_eq!(vm_val(s, "a"), int(14));
    assert_eq!(vm_val(s, "b"), int(10));
    assert_eq!(vm_val(s, "a"), val(s, "a"));
    // enum + match expression (enum patterns with binding).
    let e = "enum Sh { Circle(r) Rect(w, h) Dot }\nfn area(s) { return match s { Sh::Circle(r) => 3 * r * r, Sh::Rect(w, h) => w * h, Sh::Dot => 0 } }\nx = area(Sh::Circle(5))\ny = area(Sh::Rect(3, 4))\nz = area(Sh::Dot)";
    assert_eq!(vm_val(e, "x"), int(75));
    assert_eq!(vm_val(e, "y"), int(12));
    assert_eq!(vm_val(e, "z"), int(0));
    assert_eq!(vm_val(e, "x"), val(e, "x"));
    // statement match with struct-pattern binding.
    let m = "struct Q { a }\nq = Q { a: 7 }\nr = 0\nmatch q { Q { a } => { r = a } }";
    assert_eq!(vm_val(m, "r"), int(7));
}

#[test]
fn vm_io() {
    // cin auto-types + cout — parity (v0.24).
    let it = wide::eval_program_vm_with_input("cin >> a >> b >> c", "10 3.5 hi").unwrap();
    assert_eq!(it.get("a").unwrap(), int(10));
    assert_eq!(it.get("b").unwrap(), Value::Float(3.5));
    assert_eq!(it.get("c").unwrap(), string("hi"));
    let it2 = wide::eval_program_vm_with_input("xs = [0, 0]\ncin >> xs[0] >> xs[1]\ns = xs[0] + xs[1]", "4 5").unwrap();
    assert_eq!(it2.get("s").unwrap(), int(9));
}

#[test]
fn vm_import_modules() {
    // VM resolves imports via load_file (module flattening). The self-hosted calc runs end-to-end on the VM.
    let prog = wide::load_file(Path::new("tests/modules/calc_check.wide")).unwrap();
    let compiled = wide::compile::compile(&prog).unwrap();
    let mut machine = wide::Vm::new();
    machine.run(&compiled).unwrap();
    assert_eq!(machine.get("r1").unwrap(), int(14)); // 2 + 3*4
    assert_eq!(machine.get("r2").unwrap(), int(11)); // 2 + 3*(4-1)
    assert_eq!(machine.get("r3").unwrap(), int(21));
}

#[test]
fn vm_rejects_unsupported_clearly() {
    // Almost everything compiles now. Remaining gap: tensors (AI) — runtime error, not silently wrong.
    assert!(wide::compile_program("import \"x.wide\"").is_err(), "raw file import (must use load_file)");
    #[cfg(feature = "ai")]
    assert!(wide::eval_program_vm("import \"std/ai\"\na = tensor([1, 2, 3])").is_err(), "tensors not in VM yet");
}

// ---- v0.25: memory model ① — pointers + provenance illumination (tree-walker) ----

#[test]
fn pointers_and_provenance() {
    let it = eval_program("xs = [10, 20, 30]\np = &xs[1]\nv = *p\n*p = 99\nw = xs[1]").unwrap();
    assert_eq!(it.get("v").unwrap(), int(20)); // deref read
    assert_eq!(it.get("w").unwrap(), int(99)); // deref write visible through the array
    // provenance is illuminated (origin + extent).
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("ptr → xs[1]") && m.contains("origin xs")), "provenance: {:?}", msgs);
}

#[test]
fn pointer_errors() {
    assert!(eval_program("xs = [1]\np = &xs[5]").is_err(), "& out of bounds");
    assert!(eval_program("x = 5\ny = *x").is_err(), "deref non-pointer");
    // dangling: pointer past end after the array shrinks.
    assert!(eval_program("xs = [1, 2, 3]\np = &xs[2]\nxs.pop()\nxs.pop()\ny = *p").is_err(), "dangling deref");
}

#[test]
fn vm_pointers_match_tree_walker() {
    // v0.29: pointers (&xs[i], *p read/write) now run on the VM with the same semantics.
    assert_eq!(vm_val("xs = [10, 20, 30]\np = &xs[1]\nv = *p", "v"), int(20));
    assert_eq!(vm_val("xs = [10, 20, 30]\np = &xs[1]\n*p = 99\nw = xs[1]", "w"), int(99));
}

#[test]
fn vm_pointer_provenance_is_illuminated() {
    let vm = wide::eval_program_vm("xs = [1, 2, 3]\np = &xs[1]").unwrap();
    let msgs: Vec<String> = vm.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("ptr → xs[1]") && m.contains("extent 0..3")), "vm provenance: {:?}", msgs);
}

#[test]
fn vm_pointer_errors_match() {
    assert!(wide::eval_program_vm("xs = [1]\np = &xs[5]").is_err(), "vm & out of bounds");
    assert!(wide::eval_program_vm("x = 5\ny = *x").is_err(), "vm deref non-pointer");
}

// ---- v0.26: memory model ② — raw access + bounds illumination (tree-walker) ----

#[test]
fn raw_read_within_bounds() {
    // raw.read returns the elements; an in-bounds read is illuminated as safe (INFO, no WARN).
    let it = eval_program("xs = [10, 20, 30, 40]\nps = &xs[1]\nr = raw.read(ps, 2)\na = r[0]\nb = r[1]").unwrap();
    assert_eq!(it.get("a").unwrap(), int(20));
    assert_eq!(it.get("b").unwrap(), int(30));
    let warns = it.channel.records.iter().filter(|r| matches!(r.level, wide::lumen::Level::Warn)).count();
    assert_eq!(warns, 0, "in-bounds raw.read must not warn");
}

#[test]
fn raw_read_overrun_warns_and_clamps() {
    // A read past the extent does NOT block (unlike checked access) — it warns and clamps (honest model).
    let it = eval_program("xs = [10, 20, 30]\nps = &xs[0]\nr = raw.read(ps, 6)\nn = r.len").unwrap();
    assert_eq!(it.get("n").unwrap(), int(3), "overrun read clamps to the live buffer");
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("overrun possible") && m.contains("responsibility: caller")), "overrun WARN: {:?}", msgs);
}

#[test]
fn raw_write_through_pointer() {
    // raw.write stores values starting at the pointer; clamped past the extent.
    let it = eval_program("xs = [0, 0, 0]\np = &xs[1]\nraw.write(p, [7, 8])\na = xs[1]\nb = xs[2]").unwrap();
    assert_eq!(it.get("a").unwrap(), int(7));
    assert_eq!(it.get("b").unwrap(), int(8));
}

#[test]
fn raw_memcpy_copies_and_illuminates() {
    let it = eval_program("src = [1, 2, 3, 4]\ndst = [0, 0, 0, 0]\nps = &src[0]\npd = &dst[0]\nraw.memcpy(pd, ps, 3)\na = dst[0]\nc = dst[2]\nz = dst[3]").unwrap();
    assert_eq!(it.get("a").unwrap(), int(1));
    assert_eq!(it.get("c").unwrap(), int(3));
    assert_eq!(it.get("z").unwrap(), int(0), "only 3 elements copied");
    let warns = it.channel.records.iter().filter(|r| matches!(r.level, wide::lumen::Level::Warn)).count();
    assert_eq!(warns, 0, "in-bounds memcpy must not warn");
}

#[test]
fn raw_memcpy_overrun_warns() {
    let it = eval_program("src = [1, 2]\ndst = [0, 0, 0, 0]\nps = &src[0]\npd = &dst[0]\nraw.memcpy(pd, ps, 4)").unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("overrun possible")), "memcpy overrun WARN: {:?}", msgs);
}

#[test]
fn raw_op_errors() {
    assert!(eval_program("raw.read(5, 2)").is_err(), "raw expects a pointer");
    assert!(eval_program("xs = [1]\np = &xs[0]\nraw.bogus(p, 1)").is_err(), "unknown raw op");
}

#[test]
fn vm_raw_matches_tree_walker() {
    // v0.30: raw.read/write/memcpy now run on the VM with the same semantics + illumination.
    assert_eq!(vm_val("xs = [10, 20, 30]\nps = &xs[1]\nr = raw.read(ps, 2)\na = r[0]", "a"), int(20));
    assert_eq!(vm_val("xs = [0, 0, 0]\np = &xs[1]\nraw.write(p, [7, 8])\nb = xs[2]", "b"), int(8));
    assert_eq!(vm_val("src = [1, 2, 3, 4]\ndst = [0, 0, 0, 0]\nps = &src[0]\npd = &dst[0]\nraw.memcpy(pd, ps, 3)\nc = dst[2]", "c"), int(3));
}

#[test]
fn vm_raw_overrun_illuminated() {
    let vm = wide::eval_program_vm("xs = [1, 2, 3]\nps = &xs[0]\nr = raw.read(ps, 6)").unwrap();
    let msgs: Vec<String> = vm.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("overrun possible")), "vm raw overrun: {:?}", msgs);
}

// ---- v0.27: memory model ③ — borrow gradient (runtime guard tier, &mut XOR &) ----

#[test]
fn mutate_while_iterating_is_a_borrow_conflict() {
    // The loop holds a shared borrow of xs; pushing to xs needs exclusive access → conflict (blocked).
    match eval_program("xs = [1, 2, 3]\nfor x in xs { xs.push(x) }") {
        Err(e) => assert!(e.contains("borrow conflict"), "expected borrow conflict, got: {}", e),
        Ok(_) => panic!("mutate-while-iterating should be a borrow conflict"),
    }
}

#[test]
fn index_set_while_iterating_is_a_conflict() {
    assert!(eval_program("xs = [1, 2, 3]\nfor x in xs { xs[0] = 9 }").is_err(), "index-set while iterating");
    assert!(eval_program("xs = [1, 2, 3]\nfor x in xs { xs.sort() }").is_err(), "sort while iterating");
}

#[test]
fn shared_reads_and_other_mutations_are_fine() {
    // Reading the iterated array (shared+shared) and mutating a *different* array are allowed.
    let it = eval_program("xs = [1, 2, 3]\nys = []\nfor x in xs { ys.push(x)\n z = xs[0] }\nn = ys.len").unwrap();
    assert_eq!(it.get("n").unwrap(), int(3));
}

#[test]
fn borrow_released_after_loop() {
    // The shared borrow ends with the loop — mutating afterwards is fine again (precise liveness).
    let it = eval_program("xs = [1, 2, 3]\nfor x in xs { z = x }\nxs.push(4)\nn = xs.len").unwrap();
    assert_eq!(it.get("n").unwrap(), int(4));
}

#[test]
fn borrow_tier_is_illuminated() {
    let it = eval_program("xs = [1, 2]\nfor x in xs { z = x }").unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("shared borrow of xs") && m.contains("runtime guard")), "borrow tier: {:?}", msgs);
}

// ---- v0.28: memory model — @show provenance (unified provenance record, §3.4 / principle 5) ----

#[test]
fn show_provenance_reports_the_record() {
    let it = eval_program("xs = [10, 20, 30]\np = &xs[1]\n@show provenance p").unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(
        msgs.iter().any(|m| m.contains("provenance") && m.contains("origin xs") && m.contains("extent 0..3") && m.contains("alive true")),
        "provenance record: {:?}", msgs
    );
}

#[test]
fn show_provenance_reflects_borrow_access() {
    // The `access` field is derived from the live borrow table — shared inside an iteration.
    let it = eval_program("xs = [1, 2]\nfor x in xs { @show provenance &xs[0] }").unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("access shared(")), "borrow-aware access: {:?}", msgs);
}

#[test]
fn show_provenance_reports_dangling() {
    // After the buffer shrinks, the pointee is past the extent → alive false.
    let it = eval_program("xs = [1, 2, 3]\np = &xs[2]\nxs.pop()\nxs.pop()\n@show provenance p").unwrap();
    let msgs: Vec<String> = it.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("alive false")), "dangling provenance: {:?}", msgs);
}

#[test]
fn vm_borrow_guard_matches_tree_walker() {
    // v0.31: the borrow guard now runs on the VM — mutating the iterated array is a conflict.
    assert!(wide::eval_program_vm("xs = [1, 2, 3]\nfor x in xs { xs.push(x) }").is_err(), "vm mutate-while-iterating");
    assert!(wide::eval_program_vm("xs = [1, 2, 3]\nfor x in xs { xs[0] = 9 }").is_err(), "vm index-set while iterating");
    // ...but a different array / reads / after-loop mutation are fine, and the borrow releases.
    assert_eq!(vm_val("xs = [1, 2, 3]\nys = []\nfor x in xs { ys.push(x) }\nxs.push(4)\nn = xs.len", "n"), int(4));
}

#[test]
fn vm_show_provenance_matches_tree_walker() {
    let vm = wide::eval_program_vm("xs = [10, 20, 30]\nfor x in xs { @show provenance &xs[0] }").unwrap();
    let msgs: Vec<String> = vm.channel.records.iter().map(|r| r.msg.clone()).collect();
    assert!(msgs.iter().any(|m| m.contains("provenance") && m.contains("access shared(")), "vm @show: {:?}", msgs);
}

// ---- v0.32: static shape checking (§4.1, increment 1: literal/known shapes) ----

#[cfg(feature = "ai")]
#[test]
fn static_shape_catches_matmul_mismatch() {
    let errs = wide::type_errors("import \"std/ai\"\nc = matmul(zeros([2, 3]), zeros([4, 5]))").unwrap();
    assert!(errs.iter().any(|e| e.contains("matmul dimension mismatch") && e.contains("3≠4")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_infers_through_tensor_literals() {
    // a = (2,3), b = (2,2) → matmul inner 3 ≠ 2, caught before run.
    let errs = wide::type_errors("import \"std/ai\"\na = tensor([[1, 2, 3], [4, 5, 6]])\nb = tensor([[1, 2], [3, 4]])\nc = matmul(a, b)").unwrap();
    assert!(errs.iter().any(|e| e.contains("matmul dimension mismatch")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_catches_bad_broadcast() {
    let errs = wide::type_errors("import \"std/ai\"\nc = zeros([2, 3]) + zeros([2])").unwrap();
    assert!(errs.iter().any(|e| e.contains("not broadcastable")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_allows_valid_ops() {
    // valid matmul, valid broadcast — no false positives (principle 6).
    assert!(wide::type_errors("import \"std/ai\"\nc = matmul(zeros([2, 3]), zeros([3, 5]))").unwrap().is_empty());
    assert!(wide::type_errors("import \"std/ai\"\nc = zeros([2, 3]) + zeros([3])").unwrap().is_empty());
    assert!(wide::type_errors("import \"std/ai\"\nc = relu(matmul(zeros([4, 8]), zeros([8, 2])))").unwrap().is_empty());
}

#[cfg(feature = "ai")]
#[test]
fn static_shape_conservative_on_unknown() {
    // a parameter's shape is unknown → skipped, never a false positive.
    let errs = wide::type_errors("import \"std/ai\"\nfn f(x) { return matmul(x, zeros([9, 9])) }\nr = f(zeros([2, 3]))").unwrap();
    assert!(errs.is_empty(), "unknown shapes must be skipped: {:?}", errs);
}

// ---- v0.33: symbolic shape tier (§4.1, dimension-variable unification) ----

#[cfg(feature = "ai")]
#[test]
fn symbolic_dims_unify_consistently() {
    // M=2, K=3, N=5 — the shared K agrees; valid.
    let errs = wide::type_errors("import \"std/ai\"\na: tensor[(M, K)] = zeros([2, 3])\nb: tensor[(K, N)] = zeros([3, 5])\nc = matmul(a, b)").unwrap();
    assert!(errs.is_empty(), "consistent symbolic dims should pass: {:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn symbolic_dim_conflict_caught() {
    // shared K = 3 then 4 → caught before run via dimension-variable unification.
    let errs = wide::type_errors("import \"std/ai\"\na: tensor[(M, K)] = zeros([2, 3])\nb: tensor[(K, N)] = zeros([4, 5])").unwrap();
    assert!(errs.iter().any(|e| e.contains("dimension variable K") && e.contains("disagree")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn annotation_vs_actual_mismatch_caught() {
    let errs = wide::type_errors("import \"std/ai\"\na: tensor[(2, 3)] = zeros([2, 4])").unwrap();
    assert!(errs.iter().any(|e| e.contains("annotation says dimension 3") && e.contains("has 4")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn dynamic_and_unknown_dims_are_not_constrained() {
    // lowercase = dynamic, ? = unknown → no compile-time constraint (no false positive).
    assert!(wide::type_errors("import \"std/ai\"\na: tensor[(n, 768)] = zeros([4, 768])\nb: tensor[(?, 768)] = zeros([9, 768])").unwrap().is_empty());
}

// ---- v0.36: shape-polymorphic functions (typed params, call-site dimension unification) ----

#[cfg(feature = "ai")]
#[test]
fn shape_poly_consistent_call_passes() {
    let src = "import \"std/ai\"\nfn layer(x: tensor[(B, K)], w: tensor[(K, N)]) { return matmul(x, w) }\nr = layer(zeros([4, 8]), zeros([8, 2]))";
    assert!(wide::type_errors(src).unwrap().is_empty(), "consistent shape-poly call should pass");
}

#[cfg(feature = "ai")]
#[test]
fn shape_poly_call_site_conflict_caught() {
    // K = 8 from the first argument but 9 from the second → caught across the call boundary.
    let src = "import \"std/ai\"\nfn layer(x: tensor[(B, K)], w: tensor[(K, N)]) { return matmul(x, w) }\nr = layer(zeros([4, 8]), zeros([9, 2]))";
    let errs = wide::type_errors(src).unwrap();
    assert!(errs.iter().any(|e| e.contains("dimension variable K") && e.contains("disagree")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn shape_poly_param_literal_mismatch_caught() {
    // parameter declares (M, 3); the body matmuls against (4, 5) → inner 3 vs 4 caught in the body.
    let src = "import \"std/ai\"\nfn f(x: tensor[(M, 3)]) { return matmul(x, zeros([4, 5])) }";
    let errs = wide::type_errors(src).unwrap();
    assert!(errs.iter().any(|e| e.contains("matmul dimension mismatch")), "{:?}", errs);
}

#[cfg(feature = "ai")]
#[test]
fn untyped_params_still_unconstrained() {
    // a function with no annotations imposes no shape constraints (no false positive).
    let src = "import \"std/ai\"\nfn g(x, w) { return matmul(x, w) }\nr = g(zeros([2, 3]), zeros([9, 9]))";
    assert!(wide::type_errors(src).unwrap().is_empty(), "untyped params must not constrain");
}

#[test]
fn unknown_directive_errors() {
    assert!(wide::parse("@bogus x").is_err(), "unknown directive");
    assert!(wide::parse("@show lowering x").is_err(), "unknown @show target");
}

#[test]
fn errors_are_reported() {
    assert!(eval_program("y = x").is_err(), "undefined name");
    assert!(eval_program("y = 1 / 0").is_err(), "divide by zero");
    assert!(eval_program("y = 1 + true").is_err(), "type mismatch");
    assert!(eval_program("if 5 {\n}\n").is_err(), "condition is not bool");
    assert!(eval_program("fn f(a) { return a }\nr = f(1, 2)").is_err(), "arg count mismatch");
    assert!(eval_program("xs = [1, 2]\ny = xs[5]").is_err(), "index out of range");
    assert!(eval_program("m = map{}\ny = m[\"none\"]").is_err(), "missing key");
    assert!(eval_program("h = heap()\ny = h.pop()").is_err(), "pop from empty heap");
    assert!(eval_program("xs = []\ny = xs.pop()").is_err(), "pop from empty array");
}

//! Static checking — name resolution + arity. *Before* execution, catches undefined names, functions,
//! structs, enums, and variants, wrong argument counts, break/continue outside a loop, and return outside a function.
//!
//! wide's safety gradient (§4.5): catch only what is cheap to catch statically; values/shapes are handled at runtime + lighting.
//! Conservative — over-approximates definitions flow-insensitively to avoid false positives (every valid program passes).

use std::collections::{HashMap, HashSet};

use crate::ast::*;

pub struct CheckError {
    pub line: usize,
    pub msg: String,
}

/// Core builtins — always available (no import needed). Pub: the JIT batch (eval) also consults this —
/// builtins dispatch before user functions, so a builtin-shadowing fn must never become a JIT call target.
pub const CORE_BUILTINS: &[&str] = &[
    "print", "len", "int", "float", "str", "strbuf", "abs", "min", "max", "pow", "sqrt", "floor", "ceil",
    "hex", "bin", "ord", "chr", "assert", "err", "is_err", "err_msg",
];

/// The std module a gated builtin belongs to (kept in sync with eval.rs).
fn builtin_module(name: &str) -> Option<&'static str> {
    match name {
        "tensor" | "param" | "zeros" | "ones" | "matmul" | "conv2d" | "maxpool2d" | "relu" | "sigmoid"
        | "tanh" | "exp" | "log" | "softmax" | "transpose" | "grad_step" | "adam_step" => Some("ai"),
        "read_file" | "read_lines" | "write_file" | "append_file" | "remove_file" | "file_exists" => Some("fs"),
        "read_csv" => Some("ml"),
        "heap" => Some("heap"),
        "set" => Some("set"),
        _ => None,
    }
}

/// Collect and return all static errors in the program (does not stop at the first error).
pub fn check(prog: &[Stmt]) -> Vec<CheckError> {
    let mut c = Checker::default();
    // collect enabled std modules (import "std/X").
    for s in prog {
        if let Stmt::Import(p, _) = s {
            if let Some(m) = p.strip_prefix("std/") {
                c.enabled_std.insert(m.to_string());
            }
        }
    }
    c.collect_globals(prog);
    let empty = HashSet::new();
    c.check_stmts(prog, &empty);
    // Static shape checking (§4.1, increment 1: literal/known shapes). Only meaningful when tensors
    // can run — gated on the `ai` build feature. Catches matmul inner-dim and elementwise broadcast
    // mismatches *before* running, conservatively (unknown shapes are skipped → no false positives).
    if cfg!(feature = "ai") {
        let mut sc = ShapeChecker::default();
        sc.collect_sigs(prog);
        sc.process_stmts(prog);
        c.errors.append(&mut sc.errors);
    }
    // Borrow gradient, static-proof tier (§3.3, v0.49): definite straight-line conflicts are compile
    // errors; provably-safe claims are collected via `borrow_proofs` (the guard is skipped at runtime).
    let (mut berrs, _) = analyze_borrows(prog);
    c.errors.append(&mut berrs);
    c.errors.sort_by_key(|e| e.line);
    c.errors
}

/// The set of borrow-claim spans (line, col) proven safe at compile time — their runtime guard is
/// skipped and the proof is illuminated ("cost 0"). Consumed by the tree-walker (§3.3 static tier).
pub fn borrow_proofs(prog: &[Stmt]) -> HashSet<(usize, usize)> {
    analyze_borrows(prog).1
}

#[derive(Default)]
struct Checker {
    funcs: HashMap<String, usize>,                  // name → arity
    structs: HashMap<String, Vec<String>>,          // name → fields
    enums: HashMap<String, HashMap<String, usize>>, // name → (variant → arity)
    globals: HashSet<String>,                       // top-level variable names (flat)
    enabled_std: HashSet<String>,                   // imported std modules (ai/heap/set)
    loop_depth: usize,
    in_function: bool,
    errors: Vec<CheckError>,
}

impl Checker {
    /// Is this a callable builtin (core, or gated with its module imported and compiled in).
    fn builtin_ok(&self, name: &str) -> bool {
        if CORE_BUILTINS.contains(&name) {
            return true;
        }
        match builtin_module(name) {
            // `ai` builtins also require the `ai` build feature to be compiled in (v0.17).
            Some("ai") => cfg!(feature = "ai") && self.enabled_std.contains("ai"),
            Some(m) => self.enabled_std.contains(m),
            None => false,
        }
    }
}

impl Checker {
    fn err(&mut self, line: usize, msg: impl Into<String>) {
        self.errors.push(CheckError { line, msg: msg.into() });
    }

    fn collect_globals(&mut self, prog: &[Stmt]) {
        for s in prog {
            match s {
                Stmt::Fn { name, params, .. } => {
                    self.funcs.insert(name.clone(), params.len());
                }
                Stmt::Struct { name, fields, .. } => {
                    self.structs.insert(name.clone(), fields.clone());
                }
                Stmt::Enum { name, variants, .. } => {
                    self.enums.insert(name.clone(), variants.iter().cloned().collect());
                }
                _ => {}
            }
        }
        let mut g = HashSet::new();
        collect_bound(prog, &mut g);
        self.globals = g;
    }

    fn is_defined(&self, name: &str, local: &HashSet<String>) -> bool {
        local.contains(name)
            || self.globals.contains(name)
            || self.funcs.contains_key(name)
            || self.structs.contains_key(name)
            || self.enums.contains_key(name)
            || self.builtin_ok(name)
            || name == "map"
    }

    fn check_stmts(&mut self, stmts: &[Stmt], local: &HashSet<String>) {
        for s in stmts {
            self.check_stmt(s, local);
        }
    }

    fn check_stmt(&mut self, s: &Stmt, local: &HashSet<String>) {
        match s {
            Stmt::Let { value, .. } => self.check_expr(value, local),
            Stmt::Assign { target, value, .. } => {
                // assignment target: Ident is a definition (not checked), Index/Field check recv and idx.
                match target {
                    Expr::Ident(..) => {}
                    Expr::Index(recv, idx, _) => {
                        self.check_expr(recv, local);
                        self.check_expr(idx, local);
                    }
                    Expr::Field(recv, _, _) => self.check_expr(recv, local),
                    other => self.check_expr(other, local),
                }
                self.check_expr(value, local);
            }
            Stmt::Expr(e) => self.check_expr(e, local),
            Stmt::If { branches, else_body, .. } => {
                for (cond, body) in branches {
                    self.check_expr(cond, local);
                    self.check_stmts(body, local);
                }
                if let Some(body) = else_body {
                    self.check_stmts(body, local);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.check_expr(cond, local);
                self.loop_depth += 1;
                self.check_stmts(body, local);
                self.loop_depth -= 1;
            }
            Stmt::For { iter, body, .. } => {
                self.check_expr(iter, local);
                self.loop_depth += 1;
                self.check_stmts(body, local);
                self.loop_depth -= 1;
            }
            Stmt::Fn { params, body, .. } => {
                // function scope: globals + parameters + body bindings (flat, conservative).
                let mut fl: HashSet<String> = params.iter().cloned().collect();
                collect_bound(body, &mut fl);
                let saved = self.in_function;
                self.in_function = true;
                self.check_stmts(body, &fl);
                self.in_function = saved;
            }
            Stmt::Return(opt, span) => {
                if !self.in_function {
                    self.err(span.line, "cannot return outside a function");
                }
                if let Some(e) = opt {
                    self.check_expr(e, local);
                }
            }
            Stmt::Break(span) => {
                if self.loop_depth == 0 {
                    self.err(span.line, "cannot break outside a loop");
                }
            }
            Stmt::Continue(span) => {
                if self.loop_depth == 0 {
                    self.err(span.line, "cannot continue outside a loop");
                }
            }
            Stmt::Match { subject, arms, span } => {
                self.check_expr(subject, local);
                for arm in arms {
                    self.check_pattern(&arm.pattern, *span);
                    let mut al = local.clone();
                    collect_pattern_binds(&arm.pattern, &mut al);
                    self.check_stmts(&arm.body, &al);
                }
            }
            Stmt::Impl { methods, .. } => {
                // each method is a Stmt::Fn (first parameter self is explicit) — resolve body names just like Fn.
                for m in methods {
                    self.check_stmt(m, local);
                }
            }
            Stmt::Cout(parts, _) => {
                for e in parts {
                    self.check_expr(e, local);
                }
            }
            Stmt::Cin(targets, _) => {
                // Each target is an lvalue: Ident is a definition (not checked), Index/Field check recv/idx.
                for t in targets {
                    match t {
                        Expr::Ident(..) => {}
                        Expr::Index(recv, idx, _) => {
                            self.check_expr(recv, local);
                            self.check_expr(idx, local);
                        }
                        Expr::Field(recv, _, _) => self.check_expr(recv, local),
                        other => self.check_expr(other, local),
                    }
                }
            }
            Stmt::ShowProvenance(e, _) => self.check_expr(e, local),
            Stmt::Trust(inner, _) => self.check_stmt(inner, local), // @trust — inner statement still name-checked
            Stmt::Struct { .. } | Stmt::Enum { .. } | Stmt::Import(..) => {}
        }
    }

    fn check_expr(&mut self, e: &Expr, local: &HashSet<String>) {
        match e {
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::Str(..) | Expr::Map(_) => {}
            Expr::Ident(name, span) => {
                // a named function is also a value (first-class functions, v0.42)
                if !self.is_defined(name, local) && !self.funcs.contains_key(name) {
                    self.err(span.line, format!("undefined name '{}'", name));
                }
            }
            Expr::Array(elems, _) => {
                for el in elems {
                    self.check_expr(el, local);
                }
            }
            Expr::Range(l, r, _) | Expr::And(l, r, _) | Expr::Or(l, r, _) | Expr::Binary(_, l, r, _) => {
                self.check_expr(l, local);
                self.check_expr(r, local);
            }
            Expr::Neg(inner, _) | Expr::Not(inner, _) | Expr::Try(inner, _) | Expr::AddrOf(inner, _)
            | Expr::AddrOfMut(inner, _) | Expr::Deref(inner, _) => self.check_expr(inner, local),
            Expr::Index(recv, idx, _) => {
                self.check_expr(recv, local);
                self.check_expr(idx, local);
            }
            Expr::Call(name, args, span) => {
                for a in args {
                    self.check_expr(a, local);
                }
                if let Some(&arity) = self.funcs.get(name) {
                    if args.len() != arity {
                        self.err(span.line, format!("'{}' expects {} arg(s), got {}", name, arity, args.len()));
                    }
                } else if CORE_BUILTINS.contains(&name.as_str()) {
                    // core — always OK
                } else if let Some(m) = builtin_module(name) {
                    if m == "ai" && !cfg!(feature = "ai") {
                        self.err(span.line, format!("'{}' requires the `ai` build feature (rebuild with --features ai)", name));
                    } else if !self.enabled_std.contains(m) {
                        self.err(span.line, format!("'{}' requires `import \"std/{}\"`", name, m));
                    }
                } else if self.is_defined(name, local) {
                    // a variable may hold a function value (v0.42) — arity is a runtime concern
                } else {
                    self.err(span.line, format!("undefined function '{}'", name));
                }
            }
            Expr::Lambda(params, body, _) => {
                // like a fn body, but outer names stay visible (closures capture them)
                let mut fl: HashSet<String> = local.clone();
                fl.extend(params.iter().cloned());
                collect_bound(body, &mut fl);
                let saved = self.in_function;
                self.in_function = true;
                self.check_stmts(body, &fl);
                self.in_function = saved;
            }
            Expr::CallValue(callee, args, _) => {
                self.check_expr(callee, local);
                for a in args {
                    self.check_expr(a, local);
                }
            }
            Expr::Method(recv, _, args, _) => {
                // `raw.*` is a soft namespace, not a value (memory model, §3.2) — don't flag `raw` as undefined.
                if !matches!(recv.as_ref(), Expr::Ident(n, _) if n == "raw") {
                    self.check_expr(recv, local);
                }
                for a in args {
                    self.check_expr(a, local);
                }
            }
            Expr::Field(recv, _, _) => self.check_expr(recv, local),
            Expr::StructLit(name, fields, span) => {
                if !self.structs.contains_key(name) {
                    self.err(span.line, format!("undefined struct '{}'", name));
                }
                for (_, v) in fields {
                    self.check_expr(v, local);
                }
            }
            Expr::EnumLit(ename, variant, args, span) => {
                self.check_enum_ref(ename, variant, args.len(), span.line);
                for a in args {
                    self.check_expr(a, local);
                }
            }
            Expr::Match(subject, arms, span) => {
                self.check_expr(subject, local);
                for (pat, body) in arms {
                    self.check_pattern(pat, *span);
                    let mut al = local.clone();
                    collect_pattern_binds(pat, &mut al);
                    self.check_expr(body, &al);
                }
            }
        }
    }

    fn check_enum_ref(&mut self, ename: &str, variant: &str, arity: usize, line: usize) {
        // `Name::assoc(args)` on a class/struct is an associated-function call (v0.54) — method
        // existence is resolved like other methods (conservatively, at runtime).
        if !self.enums.contains_key(ename) && self.structs.contains_key(ename) {
            return;
        }
        match self.enums.get(ename) {
            None => self.err(line, format!("undefined enum '{}'", ename)),
            Some(vs) => match vs.get(variant) {
                None => self.err(line, format!("{} has no variant '{}'", ename, variant)),
                Some(&a) if a != arity => {
                    self.err(line, format!("{}::{} expects {} arg(s), got {}", ename, variant, a, arity))
                }
                _ => {}
            },
        }
    }

    fn check_pattern(&mut self, p: &Pattern, span: crate::span::Span) {
        match p {
            Pattern::Enum(ename, variant, subs) => {
                self.check_enum_ref(ename, variant, subs.len(), span.line);
                for s in subs {
                    self.check_pattern(s, span);
                }
            }
            Pattern::Struct(name, fields) => {
                if !self.structs.contains_key(name) {
                    self.err(span.line, format!("undefined struct '{}' (pattern)", name));
                }
                for (_, sp) in fields {
                    self.check_pattern(sp, span);
                }
            }
            _ => {}
        }
    }
}

/// Collect all variable bindings in the statements (excluding Fn bodies — those are function-local). flow-insensitive.
fn collect_bound(stmts: &[Stmt], into: &mut HashSet<String>) {
    for s in stmts {
        match s {
            Stmt::Let { name, .. } => {
                into.insert(name.clone());
            }
            Stmt::Assign { target: Expr::Ident(n, _), .. } => {
                into.insert(n.clone());
            }
            Stmt::Cin(targets, _) => {
                for t in targets {
                    if let Expr::Ident(n, _) = t {
                        into.insert(n.clone());
                    }
                }
            }
            Stmt::For { var, body, .. } => {
                into.insert(var.clone());
                collect_bound(body, into);
            }
            Stmt::While { body, .. } => collect_bound(body, into),
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    collect_bound(b, into);
                }
                if let Some(b) = else_body {
                    collect_bound(b, into);
                }
            }
            Stmt::Match { arms, .. } => {
                for arm in arms {
                    collect_pattern_binds(&arm.pattern, into);
                    collect_bound(&arm.body, into);
                }
            }
            _ => {} // Fn (locals excluded), Struct, Enum, Return, Break, Continue, Import, Expr, Assign (non-Ident)
        }
    }
}

fn collect_pattern_binds(p: &Pattern, into: &mut HashSet<String>) {
    match p {
        Pattern::Bind(n) => {
            into.insert(n.clone());
        }
        Pattern::Enum(_, _, subs) => {
            for s in subs {
                collect_pattern_binds(s, into);
            }
        }
        Pattern::Struct(_, fields) => {
            for (_, sp) in fields {
                collect_pattern_binds(sp, into);
            }
        }
        _ => {}
    }
}

// ---- static shape checking (§4.1, increment 1: literal/known shapes) ----
// A self-contained pass that tracks *statically known* tensor shapes (from literal constructors and
// shape-preserving ops) and flags definite mismatches before the program runs. Conservative: any
// shape it cannot determine is left Unknown, so a valid program never trips it (principle 6). The
// symbolic tier (tensor[f32,(M,K)] annotations + dimension-variable unification) is a later increment.

/// NumPy broadcast of two known shapes → the result shape, or None if non-broadcastable.
fn broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![0usize; n];
    for i in 0..n {
        let da = if i + a.len() < n { 1 } else { a[i + a.len() - n] };
        let db = if i + b.len() < n { 1 } else { b[i + b.len() - n] };
        out[i] = if da == db || db == 1 {
            da
        } else if da == 1 {
            db
        } else {
            return None;
        };
    }
    Some(out)
}

/// The shape of a tensor literal (tensor([[1,2,3],[4,5,6]]) -> (2,3)), if rectangular; else None.
fn literal_shape(e: &Expr) -> Option<Vec<usize>> {
    match e {
        Expr::Array(elems, _) => {
            if elems.is_empty() {
                return Some(vec![0]);
            }
            match literal_shape(&elems[0]) {
                Some(sub) => {
                    for el in &elems[1..] {
                        if literal_shape(el).as_deref() != Some(&sub) {
                            return None; // ragged -> unknown
                        }
                    }
                    let mut s = vec![elems.len()];
                    s.extend(sub);
                    Some(s)
                }
                None => {
                    if elems.iter().all(is_scalar_lit) {
                        Some(vec![elems.len()]) // a row of scalars
                    } else {
                        None
                    }
                }
            }
        }
        _ => None,
    }
}

fn is_scalar_lit(e: &Expr) -> bool {
    matches!(e, Expr::Int(..) | Expr::Float(..)) || matches!(e, Expr::Neg(inner, _) if is_scalar_lit(inner))
}

fn op_sym(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        _ => "?",
    }
}

/// Collect variable names assigned (or bound by let/for) anywhere in a body — used to conservatively
/// mark them Unknown after a branch/loop (we cannot know which path ran).
fn collect_assigned(stmts: &[Stmt], into: &mut Vec<String>) {
    for s in stmts {
        match s {
            Stmt::Let { name, .. } => into.push(name.clone()),
            Stmt::Assign { target: Expr::Ident(n, _), .. } => into.push(n.clone()),
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    collect_assigned(b, into);
                }
                if let Some(b) = else_body {
                    collect_assigned(b, into);
                }
            }
            Stmt::While { body, .. } => collect_assigned(body, into),
            Stmt::For { var, body, .. } => {
                into.push(var.clone());
                collect_assigned(body, into);
            }
            Stmt::Match { arms, .. } => {
                for arm in arms {
                    collect_assigned(&arm.body, into);
                }
            }
            _ => {}
        }
    }
}

/// A tensor dimension in the symbolic shape tier (§4.1): a literal size, a symbolic variable (`M`),
/// a dynamic dim (`n`), or fully unknown (`?`). Case carries the knowledge level (SYNTAX §4.1 rule).
#[derive(Clone, PartialEq)]
enum Dim {
    Lit(usize),
    Var(String),
    Dyn,
    Unknown,
}

impl Dim {
    fn show(&self) -> String {
        match self {
            Dim::Lit(n) => n.to_string(),
            Dim::Var(v) => v.clone(),
            Dim::Dyn => "n".to_string(),
            Dim::Unknown => "?".to_string(),
        }
    }
}

fn show_shape(s: &[Dim]) -> String {
    let parts: Vec<String> = s.iter().map(Dim::show).collect();
    format!("({})", parts.join(", "))
}

/// Parse the dims of a tensor type annotation string ("tensor[f32, (M, K)]" → [Var M, Var K]).
fn parse_tensor_annot(ty: &str) -> Option<Vec<Dim>> {
    if !ty.starts_with("tensor[") {
        return None;
    }
    let open = ty.rfind('(')?;
    let rest = &ty[open + 1..];
    let close = rest.find(')')?;
    let mut dims = Vec::new();
    for part in rest[..close].split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if p == "?" {
            dims.push(Dim::Unknown);
        } else if let Ok(n) = p.parse::<usize>() {
            dims.push(Dim::Lit(n));
        } else if p.chars().next().map_or(false, |c| c.is_ascii_uppercase()) {
            dims.push(Dim::Var(p.to_string()));
        } else {
            dims.push(Dim::Dyn);
        }
    }
    Some(dims)
}

#[derive(Default)]
struct ShapeChecker {
    shapes: HashMap<String, Vec<Dim>>,            // var -> shape (literal or symbolic); absent = unknown
    subst: HashMap<String, usize>,                // symbolic dimension variable -> concrete size, once known
    fn_sigs: HashMap<String, Vec<Option<Vec<Dim>>>>, // free function name -> per-parameter shape annotation
    errors: Vec<CheckError>,
}

impl ShapeChecker {
    fn err(&mut self, line: usize, msg: String) {
        self.errors.push(CheckError { line, msg });
    }

    /// Collect free-function shape signatures (per-parameter tensor annotations) so call sites can
    /// unify the caller's concrete argument shapes against the declared dimension variables.
    fn collect_sigs(&mut self, prog: &[Stmt]) {
        for s in prog {
            if let Stmt::Fn { name, param_types, .. } = s {
                let sig: Vec<Option<Vec<Dim>>> = param_types.iter().map(|o| o.as_deref().and_then(parse_tensor_annot)).collect();
                if sig.iter().any(|d| d.is_some()) {
                    self.fn_sigs.insert(name.clone(), sig);
                }
            }
        }
    }

    /// Unify a call's concrete argument shapes against a function's declared parameter shapes
    /// (a fresh per-call substitution). Shared dimension variables that disagree, or literal dims that
    /// mismatch, are caught across the call boundary — shape-polymorphic functions stay shape-safe.
    fn check_call_site(&mut self, fname: &str, args: &[Expr], line: usize) {
        let sig = match self.fn_sigs.get(fname) {
            Some(s) if s.len() == args.len() => s.clone(),
            _ => return,
        };
        let mut local: HashMap<String, usize> = HashMap::new();
        for (pdims_opt, arg) in sig.iter().zip(args) {
            let pdims = match pdims_opt {
                Some(p) => p,
                None => continue,
            };
            let concrete = match self.infer(arg).and_then(|s| self.resolve_all(&s)) {
                Some(c) => c,
                None => continue, // argument shape unknown — skip (conservative)
            };
            if pdims.len() != concrete.len() {
                self.err(line, format!("{}: argument has rank {} but parameter shape {} has rank {}", fname, concrete.len(), show_shape(pdims), pdims.len()));
                continue;
            }
            for (d, &av) in pdims.iter().zip(&concrete) {
                match d {
                    Dim::Lit(n) if *n != av => {
                        self.err(line, format!("{}: argument dimension {} does not match parameter dimension {}", fname, av, n));
                    }
                    Dim::Var(v) => match local.get(v) {
                        Some(&prev) if prev != av => {
                            self.err(line, format!("{}: dimension variable {} = {} from one argument but {} from another — shapes disagree", fname, v, prev, av));
                        }
                        None => {
                            local.insert(v.clone(), av);
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    }

    /// Resolve a dim to a concrete size if known (literal, or a bound symbolic variable).
    fn lit(&self, d: &Dim) -> Option<usize> {
        match d {
            Dim::Lit(n) => Some(*n),
            Dim::Var(v) => self.subst.get(v).copied(),
            _ => None,
        }
    }

    /// Resolve a whole shape to concrete sizes, if every dim is known.
    fn resolve_all(&self, s: &[Dim]) -> Option<Vec<usize>> {
        s.iter().map(|d| self.lit(d)).collect()
    }

    /// Infer the shape of an expression as dims (literal or symbolic), or None if not determinable.
    fn infer(&self, e: &Expr) -> Option<Vec<Dim>> {
        match e {
            Expr::Ident(n, _) => self.shapes.get(n).cloned(),
            Expr::Call(name, args, _) => match name.as_str() {
                // zeros/ones take one array of dims: zeros([2, 3]).
                "zeros" | "ones" => match args.first() {
                    Some(Expr::Array(elems, _)) => {
                        let mut dims = Vec::new();
                        for el in elems {
                            match el {
                                Expr::Int(n, _) if *n >= 0 => dims.push(Dim::Lit(*n as usize)),
                                _ => return None,
                            }
                        }
                        if dims.is_empty() {
                            None
                        } else {
                            Some(dims)
                        }
                    }
                    _ => None,
                },
                "tensor" | "param" => args.first().and_then(literal_shape).map(|s| s.into_iter().map(Dim::Lit).collect()),
                "transpose" => {
                    let s = self.infer(args.first()?)?;
                    if s.len() == 2 {
                        Some(vec![s[1].clone(), s[0].clone()])
                    } else {
                        None
                    }
                }
                "relu" | "sigmoid" | "tanh" | "exp" | "log" | "softmax" => self.infer(args.first()?),
                "matmul" => {
                    let a = self.infer(args.first()?)?;
                    let b = self.infer(args.get(1)?)?;
                    if a.len() == 2 && b.len() == 2 {
                        // Inner dims must agree (or be unresolved → assume ok); result is (a_rows, b_cols).
                        let ok = match (self.lit(&a[1]), self.lit(&b[0])) {
                            (Some(x), Some(y)) => x == y,
                            _ => true,
                        };
                        if ok {
                            Some(vec![a[0].clone(), b[1].clone()])
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
                "conv2d" => {
                    // Valid convolution: out dim = in − kernel + 1 per axis, when both sizes are known.
                    let a = self.infer(args.first()?)?;
                    let b = self.infer(args.get(1)?)?;
                    if a.len() == 2 && b.len() == 2 {
                        let mut out = Vec::with_capacity(2);
                        for i in 0..2 {
                            out.push(match (self.lit(&a[i]), self.lit(&b[i])) {
                                (Some(x), Some(y)) if x >= y => Dim::Lit(x - y + 1),
                                (Some(_), Some(_)) => return None, // kernel too big — check_expr reports it
                                _ => Dim::Unknown,
                            });
                        }
                        Some(out)
                    } else {
                        None
                    }
                }
                "maxpool2d" => {
                    // Non-overlapping window: out dim = in / k (floor), when the size and k are known.
                    let a = self.infer(args.first()?)?;
                    let k = match args.get(1)? {
                        Expr::Int(n, _) if *n >= 1 => *n as usize,
                        _ => return None,
                    };
                    if a.len() == 2 {
                        let mut out = Vec::with_capacity(2);
                        for d in &a {
                            out.push(match self.lit(d) {
                                Some(x) if x >= k => Dim::Lit(x / k),
                                Some(_) => return None, // window too big — check_expr reports it
                                None => Dim::Unknown,
                            });
                        }
                        Some(out)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            Expr::Binary(op, l, r, _) if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) => {
                let (a, b) = (self.infer(l)?, self.infer(r)?);
                let (ca, cb) = (self.resolve_all(&a)?, self.resolve_all(&b)?);
                broadcast(&ca, &cb).map(|v| v.into_iter().map(Dim::Lit).collect())
            }
            _ => None,
        }
    }

    /// Walk an expression, validating matmul inner-dims and elementwise broadcasts where both operand
    /// shapes resolve to concrete sizes.
    fn check_expr(&mut self, e: &Expr) {
        match e {
            Expr::Call(name, args, span) => {
                for a in args {
                    self.check_expr(a);
                }
                self.check_call_site(name, args, span.line);
                if name == "matmul" && args.len() == 2 {
                    if let (Some(a), Some(b)) = (self.infer(&args[0]), self.infer(&args[1])) {
                        if a.len() == 2 && b.len() == 2 {
                            if let (Some(x), Some(y)) = (self.lit(&a[1]), self.lit(&b[0])) {
                                if x != y {
                                    self.err(span.line, format!("matmul dimension mismatch {}·{} — inner {}≠{} (caught before run)", show_shape(&a), show_shape(&b), x, y));
                                }
                            }
                        }
                    }
                }
                if name == "conv2d" && args.len() == 2 {
                    if let (Some(a), Some(b)) = (self.infer(&args[0]), self.infer(&args[1])) {
                        if a.len() == 2 && b.len() == 2 {
                            for i in 0..2 {
                                if let (Some(x), Some(y)) = (self.lit(&a[i]), self.lit(&b[i])) {
                                    if y > x {
                                        self.err(span.line, format!("conv2d kernel {} larger than input {} (caught before run)", show_shape(&b), show_shape(&a)));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
                if name == "maxpool2d" && args.len() == 2 {
                    if let (Some(a), Some(Expr::Int(k, _))) = (self.infer(&args[0]), args.get(1)) {
                        if a.len() == 2 && *k >= 1 {
                            for d in &a {
                                if let Some(x) = self.lit(d) {
                                    if (*k as usize) > x {
                                        self.err(span.line, format!("maxpool2d window {}×{} larger than input {} (caught before run)", k, k, show_shape(&a)));
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Expr::Binary(op, l, r, span) => {
                self.check_expr(l);
                self.check_expr(r);
                if matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
                    if let (Some(a), Some(b)) = (self.infer(l), self.infer(r)) {
                        if let (Some(ca), Some(cb)) = (self.resolve_all(&a), self.resolve_all(&b)) {
                            if broadcast(&ca, &cb).is_none() {
                                self.err(span.line, format!("tensor shape mismatch {:?} {} {:?} — not broadcastable (caught before run)", ca, op_sym(op), cb));
                            }
                        }
                    }
                }
            }
            Expr::Method(recv, _, args, _) => {
                self.check_expr(recv);
                for a in args {
                    self.check_expr(a);
                }
            }
            Expr::Field(r, _, _) | Expr::Neg(r, _) | Expr::Not(r, _) | Expr::Try(r, _) | Expr::AddrOf(r, _) | Expr::Deref(r, _) => self.check_expr(r),
            Expr::Index(a, b, _) | Expr::Range(a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) => {
                self.check_expr(a);
                self.check_expr(b);
            }
            Expr::Array(elems, _) => {
                for el in elems {
                    self.check_expr(el);
                }
            }
            Expr::StructLit(_, fields, _) => {
                for (_, e) in fields {
                    self.check_expr(e);
                }
            }
            Expr::EnumLit(_, _, args, _) => {
                for a in args {
                    self.check_expr(a);
                }
            }
            Expr::Match(subj, arms, _) => {
                self.check_expr(subj);
                for (_, e) in arms {
                    self.check_expr(e);
                }
            }
            _ => {}
        }
    }

    /// Unify a declared shape annotation against the value's actual shape, binding dimension variables
    /// and reporting conflicts — the heart of the symbolic tier (§4.1: "relations checked at compile time").
    fn unify_annot(&mut self, decl: &[Dim], actual: &[Dim], line: usize) {
        if decl.len() != actual.len() {
            self.err(line, format!("shape annotation {} has rank {}, but the value has rank {}", show_shape(decl), decl.len(), actual.len()));
            return;
        }
        for (d, a) in decl.iter().zip(actual.iter()) {
            let av = match self.lit(a) {
                Some(v) => v,
                None => continue, // value's dim unknown — nothing to bind/check
            };
            match d {
                Dim::Lit(n) if *n != av => {
                    self.err(line, format!("shape annotation says dimension {} but the value has {}", n, av));
                }
                Dim::Var(v) => match self.subst.get(v) {
                    Some(&prev) if prev != av => {
                        self.err(line, format!("dimension variable {} = {} here, but it was {} earlier — shapes disagree", v, av, prev));
                    }
                    None => {
                        self.subst.insert(v.clone(), av);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    fn process_stmts(&mut self, stmts: &[Stmt]) {
        for s in stmts {
            self.process_stmt(s);
        }
    }

    /// Process a branch/loop body in an isolated env (errors still collected), then mark every variable
    /// it assigns as Unknown in the outer env — we cannot know which path ran (conservative).
    fn process_body(&mut self, body: &[Stmt]) {
        let saved_shapes = self.shapes.clone();
        let saved_subst = self.subst.clone();
        self.process_stmts(body);
        self.shapes = saved_shapes;
        self.subst = saved_subst; // branch-local dim bindings do not leak
        let mut assigned = Vec::new();
        collect_assigned(body, &mut assigned);
        for v in assigned {
            self.shapes.remove(&v);
        }
    }

    fn process_stmt(&mut self, s: &Stmt) {
        match s {
            Stmt::Let { name, ty, value, span } => {
                self.check_expr(value);
                let declared = ty.as_deref().and_then(parse_tensor_annot);
                let actual = self.infer(value);
                match declared {
                    Some(decl) => {
                        if let Some(act) = &actual {
                            self.unify_annot(&decl, act, span.line);
                        }
                        self.shapes.insert(name.clone(), decl);
                    }
                    None => match actual {
                        Some(sh) => {
                            self.shapes.insert(name.clone(), sh);
                        }
                        None => {
                            self.shapes.remove(name);
                        }
                    },
                }
            }
            Stmt::Assign { target: Expr::Ident(name, _), value, .. } => {
                self.check_expr(value);
                match self.infer(value) {
                    Some(sh) => {
                        self.shapes.insert(name.clone(), sh);
                    }
                    None => {
                        self.shapes.remove(name);
                    }
                }
            }
            Stmt::Assign { target, value, .. } => {
                self.check_expr(target);
                self.check_expr(value);
            }
            Stmt::Expr(e) | Stmt::ShowProvenance(e, _) => self.check_expr(e),
            Stmt::Return(Some(e), _) => self.check_expr(e),
            Stmt::Cout(parts, _) => {
                for e in parts {
                    self.check_expr(e);
                }
            }
            Stmt::If { branches, else_body, .. } => {
                for (cond, body) in branches {
                    self.check_expr(cond);
                    self.process_body(body);
                }
                if let Some(body) = else_body {
                    self.process_body(body);
                }
            }
            Stmt::While { cond, body, .. } => {
                self.check_expr(cond);
                self.process_body(body);
            }
            Stmt::For { iter, body, .. } => {
                self.check_expr(iter);
                self.process_body(body);
            }
            Stmt::Match { subject, arms, .. } => {
                self.check_expr(subject);
                for arm in arms {
                    self.process_body(&arm.body);
                }
            }
            Stmt::Fn { params, param_types, body, .. } => {
                // Fresh scope; seed annotated parameters with their (symbolic) shapes so the body is
                // checked against the declared dims. Dimension variables stay unbound (their values
                // arrive at call sites) — body checks only fire on definite literal conflicts.
                let saved_shapes = std::mem::take(&mut self.shapes);
                let saved_subst = std::mem::take(&mut self.subst);
                for (pname, pty) in params.iter().zip(param_types) {
                    if let Some(dims) = pty.as_deref().and_then(parse_tensor_annot) {
                        self.shapes.insert(pname.clone(), dims);
                    }
                }
                self.process_stmts(body);
                self.shapes = saved_shapes;
                self.subst = saved_subst;
            }
            Stmt::Impl { methods, .. } => {
                for m in methods {
                    self.process_stmt(m);
                }
            }
            _ => {}
        }
    }
}

// ---- borrow gradient: static-proof tier (§3.3, v0.49) ----
//
// Scope-based analysis (design agreed 2026-07-03): a claim (`r = &x` / `m = &mut x`) lives until the
// end of its enclosing block, so its *region* is the rest of that block. Three-way classification:
//   1. definite conflict — a use the runtime guard would certainly block, in *straight-line* code of
//      the same block (never nested in a conditional — those may not execute, and false positives are
//      forbidden, principle 6) → compile error, caught before run.
//   2. proven safe — every appearance of the buffer's names in the region (including nested blocks) is
//      an analyzable safe form and no conflicting form exists → the runtime guard is skipped and the
//      proof illuminated ("cost 0").
//   3. anything else (the name escapes: call argument, alias, closure capture, rebinding, @trust) →
//      the runtime-guard tier, exactly v0.48 behavior.
// The borrow binding aliases the same buffer, so uses through it count too ({x, r} alias set).

/// Analyze every block; returns (definite-conflict errors, proven claim spans).
fn analyze_borrows(prog: &[Stmt]) -> (Vec<CheckError>, HashSet<(usize, usize)>) {
    let mut errs = Vec::new();
    let mut proven = HashSet::new();
    let mut globals: HashMap<String, GlobalUse> = HashMap::new();
    global_scan(prog, false, &mut globals);
    walk_borrow_block(prog, &globals, &mut errs, &mut proven);
    (errs, proven)
}

/// Program-wide facts about one name — proofs are *sound-by-conservatism*: a claim's region analysis
/// alone cannot see claims live in outer scopes or reachable through calls, so a claim is only proven
/// when the whole program agrees (e.g. a shared claim on x needs zero `&mut x` anywhere).
#[derive(Default)]
struct GlobalUse {
    shared: usize,       // `&x` claims + `for _ in x` iterations, anywhere
    exclusive: usize,    // `&mut x` claims, anywhere
    escape: usize,       // bare uses / aliasing / call args / `@show provenance x`, anywhere
    mut_callable: usize, // mutations of x inside fn/lambda bodies (reachable during any region via a call)
}

fn walk_borrow_block(stmts: &[Stmt], globals: &HashMap<String, GlobalUse>, errs: &mut Vec<CheckError>, proven: &mut HashSet<(usize, usize)>) {
    for (i, s) in stmts.iter().enumerate() {
        // recurse into nested blocks (their own claims live in their own scopes)
        match s {
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    walk_borrow_block(b, globals, errs, proven);
                }
                if let Some(b) = else_body {
                    walk_borrow_block(b, globals, errs, proven);
                }
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } | Stmt::Fn { body, .. } => walk_borrow_block(body, globals, errs, proven),
            Stmt::Impl { methods, .. } => walk_borrow_block(methods, globals, errs, proven),
            Stmt::Match { arms, .. } => {
                for arm in arms {
                    walk_borrow_block(&arm.body, globals, errs, proven);
                }
            }
            _ => {}
        }
        // a claim at this statement?
        if let Stmt::Let { name: bvar, value, .. } | Stmt::Assign { target: Expr::Ident(bvar, _), value, .. } = s {
            let (target, excl, aspan) = match value {
                Expr::AddrOf(inner, sp) => match inner.as_ref() {
                    Expr::Ident(x, _) => (x.clone(), false, *sp),
                    _ => continue, // &xs[i] is a pointer, not a whole-buffer claim
                },
                Expr::AddrOfMut(inner, sp) => match inner.as_ref() {
                    Expr::Ident(x, _) => (x.clone(), true, *sp),
                    _ => continue,
                },
                _ => continue,
            };
            let aliases = [target.clone(), bvar.clone()];
            match classify_region(&stmts[i + 1..], &aliases, excl) {
                Region::Conflict(line, what) => errs.push(CheckError {
                    line,
                    msg: format!(
                        "borrow conflict: {} while {} is {}-borrowed (caught before run)",
                        what,
                        target,
                        if excl { "exclusively" } else { "shared" }
                    ),
                }),
                Region::Proven if globally_provable(globals, &target, bvar, excl) => {
                    proven.insert((aspan.line, aspan.col));
                }
                _ => {} // guard tier — exactly v0.48 behavior
            }
        }
    }
}

/// Whole-program side of the proof: the region can't see outer-scope claims or call-reachable uses,
/// so a claim is only proven when the name is globally quiet enough that its guard can never fire.
fn globally_provable(g: &HashMap<String, GlobalUse>, target: &str, bvar: &str, excl: bool) -> bool {
    let d = GlobalUse::default();
    let t = g.get(target).unwrap_or(&d);
    let b = g.get(bvar).unwrap_or(&d);
    if excl {
        // an exclusive claim registers only if *nothing* is live — require this to be the name's only
        // borrow-ish use anywhere, and the binder to be completely quiet.
        t.exclusive == 1 && t.shared == 0 && t.escape == 0
            && b.exclusive == 0 && b.shared == 0 && b.escape == 0
    } else {
        // a shared claim fails only against an exclusive; mutations reachable via calls could trip the
        // guard mid-region, so those must not exist either.
        t.exclusive == 0 && t.escape == 0 && t.mut_callable == 0
            && b.exclusive == 0 && b.escape == 0 && b.mut_callable == 0
    }
}

/// One pass over the whole program collecting `GlobalUse` per name. `callable` is true inside fn/impl/
/// lambda bodies (code reachable during any region via a call).
fn global_scan(stmts: &[Stmt], callable: bool, g: &mut HashMap<String, GlobalUse>) {
    let bump = |g: &mut HashMap<String, GlobalUse>, n: &str, f: fn(&mut GlobalUse)| {
        f(g.entry(n.to_string()).or_default());
    };
    fn expr(e: &Expr, callable: bool, g: &mut HashMap<String, GlobalUse>) {
        match e {
            Expr::Ident(n, _) => g.entry(n.clone()).or_default().escape += 1,
            Expr::AddrOf(inner, _) => match inner.as_ref() {
                Expr::Ident(n, _) => g.entry(n.clone()).or_default().shared += 1,
                other => expr(other, callable, g),
            },
            Expr::AddrOfMut(inner, _) => match inner.as_ref() {
                Expr::Ident(n, _) => g.entry(n.clone()).or_default().exclusive += 1,
                other => expr(other, callable, g),
            },
            Expr::Index(recv, idx, _) => {
                if !matches!(recv.as_ref(), Expr::Ident(..)) {
                    expr(recv, callable, g); // reading x[i] is safe — bare recv idents not counted
                }
                expr(idx, callable, g);
            }
            Expr::Field(recv, _, _) => {
                if !matches!(recv.as_ref(), Expr::Ident(..)) {
                    expr(recv, callable, g);
                }
            }
            Expr::Method(recv, m, args, _) => {
                if let Expr::Ident(n, _) = recv.as_ref() {
                    if callable && crate::runtime::MUT_ARRAY_METHODS.contains(&m.as_str()) {
                        g.entry(n.clone()).or_default().mut_callable += 1;
                    } // non-mut method reads are safe
                } else {
                    expr(recv, callable, g);
                }
                for a in args {
                    expr(a, callable, g);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Try(a, _) | Expr::Deref(a, _) => expr(a, callable, g),
            Expr::Range(a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) | Expr::Binary(_, a, b, _) => {
                expr(a, callable, g);
                expr(b, callable, g);
            }
            Expr::Call(_, args, _) | Expr::EnumLit(_, _, args, _) | Expr::Array(args, _) => {
                for a in args {
                    expr(a, callable, g);
                }
            }
            Expr::CallValue(c, args, _) => {
                expr(c, callable, g);
                for a in args {
                    expr(a, callable, g);
                }
            }
            Expr::StructLit(_, fields, _) => {
                for (_, v) in fields {
                    expr(v, callable, g);
                }
            }
            Expr::Match(subj, arms, _) => {
                expr(subj, callable, g);
                for (_, b) in arms {
                    expr(b, callable, g);
                }
            }
            Expr::Lambda(_, body, _) => global_scan(body, true, g),
            Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::Str(..) | Expr::Map(_) => {}
        }
    }
    for s in stmts {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { target: Expr::Ident(..), value, .. } => expr(value, callable, g),
            Stmt::Assign { target, value, .. } => {
                if let Expr::Index(recv, idx, _) = target {
                    if let Expr::Ident(n, _) = recv.as_ref() {
                        if callable {
                            bump(g, n, |u| u.mut_callable += 1);
                        }
                    } else {
                        expr(recv, callable, g);
                    }
                    expr(idx, callable, g);
                } else {
                    expr(target, callable, g);
                }
                expr(value, callable, g);
            }
            Stmt::Expr(e) | Stmt::Return(Some(e), _) => expr(e, callable, g),
            Stmt::ShowProvenance(e, _) => {
                // introspection wants the live table — a proven (unregistered) claim would be invisible,
                // so @show on a name makes claims on it guard-tier (escape).
                expr(e, callable, g);
            }
            Stmt::If { branches, else_body, .. } => {
                for (c, b) in branches {
                    expr(c, callable, g);
                    global_scan(b, callable, g);
                }
                if let Some(b) = else_body {
                    global_scan(b, callable, g);
                }
            }
            Stmt::While { cond, body, .. } => {
                expr(cond, callable, g);
                global_scan(body, callable, g);
            }
            Stmt::For { iter, body, .. } => {
                if let Expr::Ident(n, _) = iter {
                    bump(g, n, |u| u.shared += 1); // iteration registers a shared claim
                } else {
                    expr(iter, callable, g);
                }
                global_scan(body, callable, g);
            }
            Stmt::Match { subject, arms, .. } => {
                expr(subject, callable, g);
                for arm in arms {
                    global_scan(&arm.body, callable, g);
                }
            }
            Stmt::Cout(ps, _) | Stmt::Cin(ps, _) => {
                for p in ps {
                    expr(p, callable, g);
                }
            }
            Stmt::Trust(inner, _) => global_scan(std::slice::from_ref(inner), callable, g),
            Stmt::Fn { body, .. } => global_scan(body, true, g),
            Stmt::Impl { methods, .. } => global_scan(methods, true, g),
            Stmt::Struct { .. } | Stmt::Enum { .. } | Stmt::Import(..) | Stmt::Return(None, _) | Stmt::Break(_) | Stmt::Continue(_) => {}
        }
    }
}

enum Region {
    Conflict(usize, String), // definite, straight-line — compile error
    Proven,                  // safe — skip the runtime guard
    Guard,                   // undecidable — runtime-guard tier (v0.48 behavior)
}

/// How one use of an alias name relates to the claim.
#[derive(Clone, PartialEq)]
enum Use {
    Safe,      // read forms: x[i], x.len, non-mutating method
    Mutation,  // x.push(..), x[i] = v — conflicts with a *shared* claim
    AnyBorrow, // &x / `for _ in x` — conflicts with an *exclusive* claim
    MutBorrow, // &mut x — conflicts with both claim kinds
    Escape,    // the name flows somewhere we cannot analyze (call arg, alias, capture, rebinding)
}

type Uses = Vec<(Use, usize, String)>;

fn classify_region(rest: &[Stmt], aliases: &[String; 2], excl: bool) -> Region {
    let mut all: Uses = Vec::new(); // every use, anywhere in the region
    let mut straight: Uses = Vec::new(); // uses in straight-line code
    scan_stmts(rest, aliases, true, &mut all, &mut straight);
    // 1. definite conflicts in straight-line code → compile error.
    for (u, line, what) in &straight {
        let conflicts = match u {
            Use::MutBorrow => true,
            Use::AnyBorrow => excl,
            Use::Mutation => !excl,
            _ => false,
        };
        if conflicts {
            return Region::Conflict(*line, what.clone());
        }
    }
    // 2. proven: every use analyzable and non-conflicting, anywhere in the region.
    let provable = all.iter().all(|(u, _, _)| match u {
        Use::Safe => true,
        Use::Escape | Use::MutBorrow => false,
        Use::AnyBorrow => !excl,
        Use::Mutation => excl, // mutations while &mut are allowed at runtime (through the borrow)
    });
    if provable {
        Region::Proven
    } else {
        Region::Guard
    }
}

fn push_use(u: Use, line: usize, what: String, straight: bool, all: &mut Uses, st: &mut Uses) {
    if straight {
        st.push((u.clone(), line, what.clone()));
    }
    all.push((u, line, what));
}

/// Scan statements for uses of the alias names. `straight` is true for code that certainly executes
/// when reached (conditional bodies are scanned with straight=false). Scanning stops at return/break/
/// continue (code after them in the same block never runs).
fn scan_stmts(stmts: &[Stmt], aliases: &[String; 2], straight: bool, all: &mut Uses, st: &mut Uses) {
    for s in stmts {
        match s {
            Stmt::Return(opt, sp) => {
                if let Some(e) = opt {
                    scan_expr(e, aliases, straight, sp.line, all, st);
                }
                return; // the rest of this block is unreachable
            }
            Stmt::Break(_) | Stmt::Continue(_) => return,
            Stmt::Trust(_, _) => {} // @trust: checks are off inside — never a conflict; not provable either
            Stmt::Let { name, value, span, .. } | Stmt::Assign { target: Expr::Ident(name, _), value, span } => {
                scan_expr(value, aliases, straight, span.line, all, st);
                if aliases.contains(name) {
                    // rebinding an alias — we lose track of what the name means → escape
                    push_use(Use::Escape, span.line, format!("{} is rebound", name), straight, all, st);
                }
            }
            Stmt::Assign { target, value, span } => {
                // x[i] = v — mutation through an alias if the receiver is one
                if let Expr::Index(recv, idx, _) = target {
                    if let Expr::Ident(n, _) = recv.as_ref() {
                        if aliases.contains(n) {
                            push_use(Use::Mutation, span.line, format!("index assignment into {}", n), straight, all, st);
                        }
                    } else {
                        scan_expr(recv, aliases, straight, span.line, all, st);
                    }
                    scan_expr(idx, aliases, straight, span.line, all, st);
                } else {
                    scan_expr(target, aliases, straight, span.line, all, st);
                }
                scan_expr(value, aliases, straight, span.line, all, st);
            }
            Stmt::Expr(e) => {
                let line = crate::compile::expr_span(e).line;
                scan_expr(e, aliases, straight, line, all, st);
            }
            Stmt::If { branches, else_body, span } => {
                for (c, b) in branches {
                    scan_expr(c, aliases, straight, span.line, all, st);
                    scan_stmts(b, aliases, false, all, st);
                }
                if let Some(b) = else_body {
                    scan_stmts(b, aliases, false, all, st);
                }
            }
            Stmt::While { cond, body, span } => {
                scan_expr(cond, aliases, straight, span.line, all, st);
                scan_stmts(body, aliases, false, all, st);
            }
            Stmt::For { iter, body, span, .. } => {
                if let Expr::Ident(n, _) = iter {
                    if aliases.contains(n) {
                        push_use(Use::AnyBorrow, span.line, format!("iteration over {}", n), straight, all, st);
                    }
                } else {
                    scan_expr(iter, aliases, straight, span.line, all, st);
                }
                scan_stmts(body, aliases, false, all, st);
            }
            Stmt::Match { subject, arms, span } => {
                scan_expr(subject, aliases, straight, span.line, all, st);
                for arm in arms {
                    scan_stmts(&arm.body, aliases, false, all, st);
                }
            }
            Stmt::Cout(parts, span) | Stmt::Cin(parts, span) => {
                for p in parts {
                    scan_expr(p, aliases, straight, span.line, all, st);
                }
            }
            Stmt::ShowProvenance(e, span) => {
                // introspection only reads the borrow table — safe, whatever the operand
                if !matches!(e, Expr::Ident(n, _) if aliases.contains(n)) {
                    scan_expr(e, aliases, straight, span.line, all, st);
                }
            }
            Stmt::Fn { .. } | Stmt::Impl { .. } | Stmt::Struct { .. } | Stmt::Enum { .. } | Stmt::Import(..) => {}
        }
    }
}

/// Classify uses of the alias names inside one expression.
fn scan_expr(e: &Expr, aliases: &[String; 2], straight: bool, line: usize, all: &mut Uses, st: &mut Uses) {
    match e {
        Expr::Ident(n, sp) => {
            if aliases.contains(n) {
                // a bare appearance — the value flows somewhere (alias/arg/return) → unanalyzable
                push_use(Use::Escape, sp.line, format!("{} escapes", n), straight, all, st);
            }
        }
        Expr::AddrOf(inner, sp) => match inner.as_ref() {
            Expr::Ident(n, _) if aliases.contains(n) => push_use(Use::AnyBorrow, sp.line, format!("& of {}", n), straight, all, st),
            _ => scan_expr(inner, aliases, straight, line, all, st),
        },
        Expr::AddrOfMut(inner, sp) => match inner.as_ref() {
            Expr::Ident(n, _) if aliases.contains(n) => push_use(Use::MutBorrow, sp.line, format!("&mut of {}", n), straight, all, st),
            _ => scan_expr(inner, aliases, straight, line, all, st),
        },
        Expr::Index(recv, idx, _) => {
            // reading x[i] is a shared access — safe at runtime for both claim kinds
            if let Expr::Ident(n, sp) = recv.as_ref() {
                if aliases.contains(n) {
                    push_use(Use::Safe, sp.line, format!("read {}[..]", n), straight, all, st);
                }
            } else {
                scan_expr(recv, aliases, straight, line, all, st);
            }
            scan_expr(idx, aliases, straight, line, all, st);
        }
        Expr::Field(recv, _, _) => {
            // x.len etc. — a read
            if let Expr::Ident(n, sp) = recv.as_ref() {
                if aliases.contains(n) {
                    push_use(Use::Safe, sp.line, format!("read field of {}", n), straight, all, st);
                }
            } else {
                scan_expr(recv, aliases, straight, line, all, st);
            }
        }
        Expr::Method(recv, m, args, sp) => {
            if let Expr::Ident(n, _) = recv.as_ref() {
                if aliases.contains(n) {
                    if crate::runtime::MUT_ARRAY_METHODS.contains(&m.as_str()) {
                        push_use(Use::Mutation, sp.line, format!("{}.{}()", n, m), straight, all, st);
                    } else {
                        push_use(Use::Safe, sp.line, format!("read method {}.{}()", n, m), straight, all, st);
                    }
                }
            } else {
                scan_expr(recv, aliases, straight, line, all, st);
            }
            for a in args {
                scan_expr(a, aliases, straight, line, all, st);
            }
        }
        Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Try(a, _) | Expr::Deref(a, _) => scan_expr(a, aliases, straight, line, all, st),
        Expr::Range(a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) | Expr::Binary(_, a, b, _) => {
            scan_expr(a, aliases, straight, line, all, st);
            scan_expr(b, aliases, straight, line, all, st);
        }
        Expr::Call(_, args, _) | Expr::EnumLit(_, _, args, _) | Expr::Array(args, _) => {
            for a in args {
                scan_expr(a, aliases, straight, line, all, st);
            }
        }
        Expr::CallValue(c, args, _) => {
            scan_expr(c, aliases, straight, line, all, st);
            for a in args {
                scan_expr(a, aliases, straight, line, all, st);
            }
        }
        Expr::StructLit(_, fields, _) => {
            for (_, v) in fields {
                scan_expr(v, aliases, straight, line, all, st);
            }
        }
        Expr::Match(subj, arms, _) => {
            scan_expr(subj, aliases, straight, line, all, st);
            for (_, b) in arms {
                scan_expr(b, aliases, false, line, all, st);
            }
        }
        Expr::Lambda(_, body, sp) => {
            // if the lambda body mentions an alias, the buffer escapes into the closure
            let mut used = std::collections::HashSet::new();
            crate::eval::collect_free_idents(body, &mut used);
            if aliases.iter().any(|a| used.contains(a.as_str())) {
                push_use(Use::Escape, sp.line, "captured by a closure".to_string(), straight, all, st);
            }
        }
        Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::Str(..) | Expr::Map(_) => {}
    }
}

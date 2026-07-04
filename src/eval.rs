//! Tree-walking evaluator. Emits values + illumination together — visible cost from line 1.
//!
//! Scopes: globals + function frames (locals stack). Functions can't see the caller's locals (lexical).
//! Control flow: Flow(Normal/Return/Break/Continue) propagates upward through statement execution.
//! Collections: Rc<RefCell> reference semantics — push/pop/sort operate in place.

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::ast::*;
use crate::lumen::Channel;
use crate::span::Span;
#[cfg(feature = "ai")]
use std::rc::Rc;

use crate::value::{value_cmp, ArrayRef, HeapRef, MapRef, PtrData, SetRef, StrBufRef, Value};
#[cfg(feature = "ai")]
use crate::value::{Device, GradNode, GradOp, TensorData, TensorRef};

/// When `?` hits an error, this sentinel is sent through the error channel to auto-propagate
/// up to the nearest function-call boundary (Rust `?` bubbling). `call` intercepts it and turns it back into the error value.
/// Real error messages never contain `\u{1}`, so there's no collision.
const PROPAGATE: &str = "\u{1}__wide_propagate__";

#[derive(Clone)]
struct Func {
    params: Vec<String>,
    body: Vec<Stmt>,
}

enum Flow {
    Normal,
    Return(Value),
    Break,
    Continue,
}

pub struct Interp {
    globals: HashMap<String, Value>,
    locals: Vec<HashMap<String, Value>>,
    funcs: HashMap<String, Func>,
    methods: HashMap<String, HashMap<String, Func>>,  // struct name → (method name → Func), explicit self
    structs: HashMap<String, Vec<String>>,            // name → field names
    enums: HashMap<String, HashMap<String, usize>>,   // name → (variant → arity)
    err_value: Option<Value>,                         // error value being propagated by `?`
    std_modules: std::collections::HashSet<String>,   // activated std modules (ai/heap/set)
    input: std::collections::VecDeque<String>,        // pending whitespace-split tokens for `cin`
    stdin_enabled: bool,                              // when input is empty, refill from real stdin (false in tests)
    // Live borrows on array buffers (memory model §3.3 borrow gradient, runtime-guard tier).
    // Keyed by buffer identity (Rc address). `for x in xs` and explicit `&x` hold shared claims;
    // `&mut x` holds an exclusive claim (v0.48). Mutation-while-shared and borrow-vs-borrow conflict.
    borrows: crate::runtime::Borrows,
    // Explicit borrow claims tied to their creating scope: (frame id, scope depth, buffer, exclusive).
    // Released when that scope pops (or the owning function frame returns) — scope-based lifetime (v0.48).
    scope_claims: Vec<(usize, usize, usize, bool)>,
    frame_id: usize,   // current function frame (0 = top level)
    next_frame: usize, // fresh frame ids for calls
    trusted: usize,    // >0 inside `@trust` — borrow registration/checks are off (WARN illuminated)
    // Claims proven safe at compile time (v0.49, check::borrow_proofs) — their runtime guard is
    // skipped and the proof illuminated ("cost 0"). Keyed by the claim expression's (line, col).
    proven: std::collections::HashSet<(usize, usize)>,
    #[cfg(feature = "jit")]
    jit: Option<crate::jit::Jit>, // Cranelift JIT module (lazily created), stage 3
    #[cfg(feature = "jit")]
    jit_fns: HashMap<String, crate::jit::Compiled>, // function name -> native code (integer subset)
    pub channel: Channel,
}


/// The std module a builtin belongs to (None = core, always available).
fn builtin_module(name: &str) -> Option<&'static str> {
    match name {
        "tensor" | "param" | "zeros" | "ones" | "matmul" | "conv2d" | "maxpool2d" | "relu" | "sigmoid"
        | "tanh" | "exp" | "log" | "softmax" | "transpose" | "grad_step" | "adam_step" => Some("ai"),
        "read_file" | "read_lines" | "write_file" | "append_file" | "remove_file" | "file_exists" => Some("fs"),
        "heap" => Some("heap"),
        "set" => Some("set"),
        _ => None,
    }
}

impl Interp {
    pub fn new() -> Self {
        Interp {
            globals: HashMap::new(),
            locals: Vec::new(),
            funcs: HashMap::new(),
            methods: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            err_value: None,
            std_modules: std::collections::HashSet::new(),
            input: std::collections::VecDeque::new(),
            stdin_enabled: true,
            borrows: HashMap::new(),
            scope_claims: Vec::new(),
            frame_id: 0,
            next_frame: 1,
            trusted: 0,
            proven: Default::default(),
            #[cfg(feature = "jit")]
            jit: None,
            #[cfg(feature = "jit")]
            jit_fns: HashMap::new(),
            channel: Channel::new(),
        }
    }

    /// Feed `cin` from a string instead of real stdin (for tests / embedding).
    /// Tokens are whitespace-split; real stdin is disabled.
    pub fn set_input(&mut self, s: &str) {
        self.input = s.split_whitespace().map(|t| t.to_string()).collect();
        self.stdin_enabled = false;
    }

    /// Next whitespace-delimited input token for `cin` (refills from stdin on demand).
    fn next_input_token(&mut self) -> Option<String> {
        loop {
            if let Some(t) = self.input.pop_front() {
                return Some(t);
            }
            if !self.stdin_enabled {
                return None;
            }
            let mut line = String::new();
            match std::io::stdin().read_line(&mut line) {
                Ok(0) | Err(_) => return None, // EOF or error
                Ok(_) => {
                    for tok in line.split_whitespace() {
                        self.input.push_back(tok.to_string());
                    }
                }
            }
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        for s in self.locals.iter().rev() {
            if let Some(v) = s.get(name) {
                return Some(v.clone());
            }
        }
        self.globals.get(name).cloned()
    }

    fn define(&mut self, name: &str, val: Value) {
        if let Some(top) = self.locals.last_mut() {
            top.insert(name.to_string(), val);
        } else {
            self.globals.insert(name.to_string(), val);
        }
    }

    fn assign(&mut self, name: &str, val: Value) {
        for s in self.locals.iter_mut().rev() {
            if s.contains_key(name) {
                s.insert(name.to_string(), val);
                return;
            }
        }
        if self.globals.contains_key(name) {
            self.globals.insert(name.to_string(), val);
            return;
        }
        self.define(name, val);
    }

    /// Assign a value to an lvalue expression (Ident / Index / Field). Shared by `=` and `cin >>`.
    fn assign_lvalue(&mut self, target: &Expr, v: Value, span: Span) -> Result<(), String> {
        match target {
            Expr::Ident(name, _) => self.assign(name, v),
            Expr::Index(recv, idx, ispan) => {
                let r = self.eval(recv)?;
                let i = self.eval(idx)?;
                self.index_set(r, i, v, *ispan)?;
            }
            Expr::Field(recv, fname, fspan) => {
                let r = self.eval(recv)?;
                match &r {
                    Value::Struct { fields, name } => {
                        if !fields.borrow().contains_key(fname) {
                            return Err(format!("line {}: {} has no field '{}'", fspan.line, name, fname));
                        }
                        fields.borrow_mut().insert(fname.clone(), v);
                    }
                    other => return Err(format!("line {}: cannot assign field on {}", fspan.line, other.type_name())),
                }
            }
            Expr::Deref(inner, dspan) => {
                let p = self.deref_ptr(inner, *dspan)?;
                crate::runtime::deref_write(&p, v, *dspan)?;
            }
            _ => return Err(format!("line {}: invalid assignment target", span.line)),
        }
        Ok(())
    }

    /// Take the address of an lvalue → a pointer with provenance (memory model, §3.1). Arrays element only (v0.25).
    fn addr_of(&mut self, inner: &Expr, span: Span) -> Result<Value, String> {
        match inner {
            Expr::Index(arr_e, idx_e, _) => {
                let arr = match self.eval(arr_e)? {
                    Value::Array(a) => a,
                    other => return Err(format!("line {}: can only take & of an array element yet ({})", span.line, other.type_name())),
                };
                let idx = match self.eval(idx_e)? {
                    Value::Int(n) if n >= 0 => n as usize,
                    other => return Err(format!("line {}: pointer index must be a non-negative int ({})", span.line, other.type_name())),
                };
                let name = match arr_e.as_ref() {
                    Expr::Ident(n, _) => n.clone(),
                    _ => "?".to_string(),
                };
                crate::runtime::make_ptr(arr, idx, name, span, &mut self.channel)
            }
            // `&x` on a whole array variable — a *shared borrow claim* for the current scope (v0.48,
            // §3.3 runtime-guard tier). The value is the same buffer (borrows are tracked aliases —
            // everything in wide is already a reference; & adds the discipline + illumination).
            Expr::Ident(name, _) => {
                let v = self.eval(inner)?;
                let a = match &v {
                    Value::Array(a) => a.clone(),
                    other => return Err(format!("line {}: & takes an array variable or element (found {})", span.line, other.type_name())),
                };
                self.claim(&a, name, false, span)?;
                Ok(v)
            }
            other => {
                let _ = self.eval(other); // surface inner errors
                Err(format!("line {}: & takes an array element (&xs[i]) or an array variable (&xs)", span.line))
            }
        }
    }

    /// Register an explicit borrow claim tied to the current scope (v0.48). `@trust` skips it (WARN);
    /// a claim proven safe at compile time (v0.49) skips the guard and illuminates the proof — cost 0.
    fn claim(&mut self, a: &crate::value::ArrayRef, name: &str, exclusive: bool, span: Span) -> Result<(), String> {
        if self.proven.contains(&(span.line, span.col)) {
            let kind = if exclusive { "&mut exclusive" } else { "shared" };
            self.channel.info(span, format!("{} borrow of {} statically proven safe — cost 0 (no runtime guard)", kind, name));
            return Ok(());
        }
        if self.trusted > 0 {
            self.channel.warn(span, format!("trusted borrow of {} — checks off (responsibility: caller)", name));
            return Ok(());
        }
        if exclusive {
            crate::runtime::borrow_register_mut(&mut self.borrows, a, name, span, &mut self.channel)?;
        } else {
            crate::runtime::borrow_register(&mut self.borrows, a, name, "(&)", span, &mut self.channel)?;
        }
        self.scope_claims.push((self.frame_id, self.locals.len(), crate::runtime::buf_id(a), exclusive));
        Ok(())
    }

    fn deref_ptr(&mut self, inner: &Expr, span: Span) -> Result<std::rc::Rc<PtrData>, String> {
        match self.eval(inner)? {
            Value::Ptr(p) => Ok(p),
            other => Err(format!("line {}: cannot dereference {} (not a pointer)", span.line, other.type_name())),
        }
    }

    // ---- raw namespace (memory model, §3.2 / §3.5 / C6) ----
    // Low-level ops that *illuminate* bounds instead of blocking them. Tracking stays on: every raw
    // access is compared against the pointer's provenance extent and the result is reported as INFO
    // (within bounds, safe) or WARN (overrun possible — responsibility shifts to the caller). This is
    // the memory side of D-novelty: "a check that blocks vs a check that illuminates" (§3.2).
    //
    // Honesty (§3.7): the interpreter has no real addresses, so an overrun cannot actually corrupt a
    // neighbouring allocation — the bounds *illumination* is real, the silicon overrun is modeled (the
    // ops clamp to the live buffer). Extent is counted in *elements*, not bytes, because a wide buffer
    // is a `Vec<Value>` here (reporting bytes would be a fiction).

    // ---- raw namespace (memory model, §3.2/§3.5/C6) — semantics live in `runtime::raw_op` (shared with the VM) ----
    fn raw_op(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Value, String> {
        let vals = self.eval_args(args)?;
        crate::runtime::raw_op(name, &vals, span, &mut self.channel)
    }

    fn push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        // Scope-based borrow lifetime (v0.48): explicit claims made in this scope end here.
        let depth = self.locals.len();
        while let Some(&(fid, d, buf, excl)) = self.scope_claims.last() {
            if fid != self.frame_id || d != depth {
                break;
            }
            self.scope_claims.pop();
            if excl {
                crate::runtime::borrow_release_mut(&mut self.borrows, buf);
            } else {
                crate::runtime::borrow_release(&mut self.borrows, buf);
            }
        }
        self.locals.pop();
    }

    /// Release every explicit borrow claim made inside the given function frame (v0.48) — called when
    /// the frame returns, whatever the exit path.
    fn release_frame_claims(&mut self, fid: usize) {
        while let Some(&(f, _, buf, excl)) = self.scope_claims.last() {
            if f != fid {
                break;
            }
            self.scope_claims.pop();
            if excl {
                crate::runtime::borrow_release_mut(&mut self.borrows, buf);
            } else {
                crate::runtime::borrow_release(&mut self.borrows, buf);
            }
        }
    }

    pub fn run(&mut self, prog: &[Stmt]) -> Result<(), String> {
        // Borrow static-proof tier (v0.49): claims proven safe at compile time skip their runtime guard.
        self.proven = crate::check::borrow_proofs(prog);
        // Hoist function/struct/enum definitions — order-independent, mutual recursion allowed.
        #[cfg(feature = "jit")]
        let mut jit_cands: Vec<(String, Vec<String>, Vec<Stmt>, Span)> = Vec::new();
        for s in prog {
            match s {
                Stmt::Fn { name, params, body, span, .. } => {
                    self.funcs.insert(
                        name.clone(),
                        Func { params: params.clone(), body: body.clone() },
                    );
                    #[cfg(feature = "jit")]
                    jit_cands.push((name.clone(), params.clone(), body.clone(), *span));
                    #[cfg(not(feature = "jit"))]
                    let _ = span;
                }
                Stmt::Struct { name, fields, .. } => {
                    self.structs.insert(name.clone(), fields.clone());
                }
                Stmt::Enum { name, variants, .. } => {
                    self.enums
                        .insert(name.clone(), variants.iter().cloned().collect());
                }
                Stmt::Impl { type_name, methods, .. } => {
                    let table = self.methods.entry(type_name.clone()).or_default();
                    for m in methods {
                        if let Stmt::Fn { name, params, body, .. } = m {
                            table.insert(
                                name.clone(),
                                Func { params: params.clone(), body: body.clone() },
                            );
                        }
                    }
                }
                Stmt::Import(p, _) => {
                    if let Some(m) = p.strip_prefix("std/") {
                        self.std_modules.insert(m.to_string());
                    }
                }
                _ => {}
            }
        }
        // JIT the eligible subset as a batch — declared together so they can call each other (v0.41).
        #[cfg(feature = "jit")]
        self.try_jit_batch(jit_cands);
        match self.exec_stmts(prog) {
            Ok(Flow::Normal) => Ok(()),
            Ok(Flow::Return(_)) => Err("cannot return at top level".into()),
            Ok(Flow::Break) | Ok(Flow::Continue) => Err("cannot break/continue outside a loop".into()),
            Err(ref s) if s == PROPAGATE => Err(format!(
                "error propagated to top level: {}",
                self.err_value.take().map(|v| v.to_string()).unwrap_or_default()
            )),
            Err(e) => Err(e),
        }
    }

    fn exec_stmts(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        for s in stmts {
            match self.exec(s)? {
                Flow::Normal => {}
                other => return Ok(other),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_block(&mut self, stmts: &[Stmt]) -> Result<Flow, String> {
        self.push_scope();
        let r = self.exec_stmts(stmts);
        self.pop_scope();
        r
    }

    fn exec(&mut self, s: &Stmt) -> Result<Flow, String> {
        match s {
            Stmt::Let { name, ty, value, .. } => {
                let v = self.eval(value)?;
                if ty.is_some() {
                    self.define(name, v);
                } else {
                    self.assign(name, v);
                }
                Ok(Flow::Normal)
            }
            Stmt::Assign { target, value, span } => {
                let v = self.eval(value)?;
                self.assign_lvalue(target, v, *span)?;
                Ok(Flow::Normal)
            }
            Stmt::Expr(e) => {
                self.eval(e)?;
                Ok(Flow::Normal)
            }
            Stmt::Fn { .. } | Stmt::Struct { .. } | Stmt::Enum { .. } | Stmt::Impl { .. } => Ok(Flow::Normal),
            Stmt::Import(p, span) => {
                if p.starts_with("std/") {
                    Ok(Flow::Normal) // std modules are activated during hoisting
                } else {
                    Err(format!("line {}: unresolved import '{}' — files only via load_file", span.line, p))
                }
            }
            Stmt::Cout(parts, _) => {
                // Stream output (C++ style): no auto-space, no trailing newline — use "\n" explicitly.
                use std::io::Write;
                let mut out = String::new();
                for e in parts {
                    out.push_str(&self.eval(e)?.to_string());
                }
                print!("{}", out);
                let _ = std::io::stdout().flush();
                Ok(Flow::Normal)
            }
            Stmt::Cin(targets, span) => {
                // Read one whitespace-delimited token per target, auto-typed (int→float→str), assign to the lvalue.
                for t in targets {
                    let tok = self
                        .next_input_token()
                        .ok_or_else(|| format!("line {}: cin — no more input", span.line))?;
                    self.channel.info(*span, format!("stdin read: \"{}\"", tok));
                    let v = parse_input_token(&tok);
                    self.assign_lvalue(t, v, *span)?;
                }
                Ok(Flow::Normal)
            }
            Stmt::ShowProvenance(e, span) => {
                let v = self.eval(e)?;
                let record = crate::runtime::provenance_record(&self.borrows, &v);
                self.channel.info(*span, record);
                Ok(Flow::Normal)
            }
            Stmt::Trust(inner, span) => {
                // The trust tier of the gradient (v0.48): borrow registration/checks off for this
                // statement — illuminated as WARN, responsibility moves to the writer (§3.3).
                self.channel.warn(*span, "@trust — borrow checks off for this statement (responsibility: writer)");
                self.trusted += 1;
                let r = self.exec(inner);
                self.trusted -= 1;
                r
            }
            Stmt::Match { subject, arms, span } => {
                let val = self.eval(subject)?;
                for arm in arms {
                    if let Some(bindings) = match_pattern(&arm.pattern, &val) {
                        self.push_scope();
                        for (n, v) in bindings {
                            self.define(&n, v);
                        }
                        let r = self.exec_stmts(&arm.body);
                        self.pop_scope();
                        return r;
                    }
                }
                Err(format!("line {}: match failed — no matching arm ({})", span.line, val))
            }
            Stmt::Return(opt, _) => {
                let v = match opt {
                    Some(e) => self.eval(e)?,
                    None => Value::Unit,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Break(_) => Ok(Flow::Break),
            Stmt::Continue(_) => Ok(Flow::Continue),
            Stmt::If { branches, else_body, span } => {
                for (cond, body) in branches {
                    if self.cond(cond, *span)? {
                        return self.exec_block(body);
                    }
                }
                if let Some(body) = else_body {
                    return self.exec_block(body);
                }
                Ok(Flow::Normal)
            }
            Stmt::While { cond, body, span } => {
                while self.cond(cond, *span)? {
                    match self.exec_block(body)? {
                        Flow::Break => break,
                        Flow::Continue | Flow::Normal => {}
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For { var, iter, body, span } => self.exec_for(var, iter, body, *span),
        }
    }

    fn exec_for(&mut self, var: &str, iter: &Expr, body: &[Stmt], span: Span) -> Result<Flow, String> {
        let iter_val = self.eval(iter)?;
        // Borrow gradient (§3.3, runtime-guard tier): iterating an array holds a *shared* borrow of
        // its buffer for the loop body. Mutating that same buffer inside the loop is a `&mut`-while-`&`
        // conflict (iterator invalidation — C++ UB, a footgun elsewhere). Liveness is exactly the loop,
        // so this is false-positive-free (principle 6). The guard is released on every exit path below.
        let guard = if let Value::Array(a) = &iter_val {
            let name = if let Expr::Ident(n, _) = iter { n.clone() } else { "array".to_string() };
            let id = crate::runtime::buf_id(a);
            // Iterating while `&mut`-borrowed is a & vs &mut conflict (v0.48) — register is checked now.
            crate::runtime::borrow_register(&mut self.borrows, a, &name, "for iteration", span, &mut self.channel)?;
            Some(id)
        } else {
            None
        };
        let result = self.run_for_items(var, iter_val, body, span);
        if let Some(id) = guard {
            crate::runtime::borrow_release(&mut self.borrows, id);
        }
        result
    }

    fn run_for_items(&mut self, var: &str, iter_val: Value, body: &[Stmt], span: Span) -> Result<Flow, String> {
        let items = self.items_of(iter_val, span)?;
        for item in items {
            self.push_scope();
            self.define(var, item);
            let flow = self.exec_stmts(body);
            self.pop_scope();
            match flow? {
                Flow::Break => break,
                Flow::Continue | Flow::Normal => {}
                Flow::Return(v) => return Ok(Flow::Return(v)),
            }
        }
        Ok(Flow::Normal)
    }

    fn items_of(&mut self, v: Value, span: Span) -> Result<Vec<Value>, String> {
        match v {
            Value::Array(a) => Ok(a.borrow().clone()),
            Value::Range(a, b) => Ok((a..b).map(Value::Int).collect()),
            Value::Str(s) => Ok(s.chars().map(|c| Value::Str(c.to_string())).collect()),
            other => Err(format!("line {}: {} is not iterable (array, range, or string only)", span.line, other.type_name())),
        }
    }

    fn cond(&mut self, e: &Expr, span: Span) -> Result<bool, String> {
        let v = self.eval(e)?;
        v.truthy()
            .ok_or_else(|| format!("line {}: condition must be bool (found {})", span.line, v.type_name()))
    }

    fn eval(&mut self, e: &Expr) -> Result<Value, String> {
        match e {
            Expr::Int(n, _) => Ok(Value::Int(*n)),
            Expr::Float(x, _) => Ok(Value::Float(*x)),
            Expr::Bool(b, _) => Ok(Value::Bool(*b)),
            Expr::Str(s, _) => Ok(Value::Str(s.clone())),
            Expr::Ident(name, span) => match self.get(name) {
                Some(v) => Ok(v),
                // a named function referenced as a value (first-class functions, v0.42)
                None => match self.funcs.get(name) {
                    Some(f) => Ok(Value::Fn(std::rc::Rc::new(crate::value::FnData {
                        name: Some(name.clone()),
                        params: f.params.clone(),
                        body: f.body.clone(),
                        captured: HashMap::new(),
                    }))),
                    None => Err(format!("line {}: undefined name '{}'", span.line, name)),
                },
            },
            Expr::Array(elems, span) => {
                let mut xs = Vec::new();
                for el in elems {
                    xs.push(self.eval(el)?);
                }
                self.channel
                    .info(*span, format!("heap vector · {} elems (mutable, shared ref)", xs.len()));
                Ok(Value::array(xs))
            }
            Expr::Map(span) => {
                self.channel.info(*span, "heap map (mutable, dynamic size)");
                Ok(Value::empty_map())
            }
            Expr::Range(l, r, span) => {
                let a = self.as_int(l, *span)?;
                let b = self.as_int(r, *span)?;
                Ok(Value::Range(a, b))
            }
            Expr::Index(recv, idx, span) => {
                let r = self.eval(recv)?;
                let i = self.eval(idx)?;
                self.index_get(r, i, *span)
            }
            Expr::StructLit(name, fields, span) => self.struct_lit(name, fields, *span),
            Expr::EnumLit(ename, variant, args, span) => self.enum_lit(ename, variant, args, *span),
            Expr::Match(subject, arms, span) => {
                let val = self.eval(subject)?;
                for (pat, body) in arms {
                    if let Some(binds) = match_pattern(pat, &val) {
                        self.push_scope();
                        for (n, v) in binds {
                            self.define(&n, v);
                        }
                        let r = self.eval(body);
                        self.pop_scope();
                        return r;
                    }
                }
                Err(format!("line {}: match expression failed — no matching arm ({})", span.line, val))
            }
            Expr::Try(inner, _span) => {
                let v = self.eval(inner)?;
                if matches!(v, Value::Err(_)) {
                    // On error, return that error value from the current function (bubble via sentinel).
                    self.err_value = Some(v);
                    return Err(PROPAGATE.to_string());
                }
                Ok(v)
            }
            Expr::Neg(inner, span) => match self.eval(inner)? {
                Value::Int(n) => Ok(Value::Int(-n)),
                Value::Float(x) => Ok(Value::Float(-x)),
                v => Err(format!("line {}: cannot apply '-' to {}", span.line, v.type_name())),
            },
            Expr::Not(inner, span) => {
                let v = self.eval(inner)?;
                match v.truthy() {
                    Some(b) => Ok(Value::Bool(!b)),
                    None => Err(format!("line {}: 'not' applies to bool only (found {})", span.line, v.type_name())),
                }
            }
            Expr::AddrOf(inner, span) => self.addr_of(inner, *span),
            Expr::AddrOfMut(inner, span) => {
                // `&mut x` — an exclusive borrow claim for the current scope (v0.48, §3.3).
                let name = match inner.as_ref() {
                    Expr::Ident(n, _) => n.clone(),
                    _ => return Err(format!("line {}: &mut takes an array variable (e.g. &mut xs)", span.line)),
                };
                let v = self.eval(inner)?;
                let a = match &v {
                    Value::Array(a) => a.clone(),
                    other => return Err(format!("line {}: &mut takes an array variable (found {})", span.line, other.type_name())),
                };
                self.claim(&a, &name, true, *span)?;
                Ok(v)
            }
            Expr::Deref(inner, span) => {
                let p = self.deref_ptr(inner, *span)?;
                crate::runtime::deref_read(&p, *span)
            }
            Expr::And(l, r, span) => {
                let lv = self.eval(l)?;
                match lv.truthy() {
                    Some(false) => Ok(Value::Bool(false)),
                    Some(true) => self.bool_operand(r, "and", *span),
                    None => Err(format!("line {}: 'and' operand must be bool (found {})", span.line, lv.type_name())),
                }
            }
            Expr::Or(l, r, span) => {
                let lv = self.eval(l)?;
                match lv.truthy() {
                    Some(true) => Ok(Value::Bool(true)),
                    Some(false) => self.bool_operand(r, "or", *span),
                    None => Err(format!("line {}: 'or' operand must be bool (found {})", span.line, lv.type_name())),
                }
            }
            Expr::Binary(op, l, r, span) => {
                let lv = self.eval(l)?;
                let rv = self.eval(r)?;
                self.binop(op, lv, rv, *span)
            }
            Expr::Call(name, args, span) => self.call(name, args, *span),
            Expr::Method(recv, name, args, span) => {
                // raw.* soft namespace — recognised only when `raw` is not a bound value (memory model, §3.2 / C6).
                if let Expr::Ident(id, _) = recv.as_ref() {
                    if id == "raw" && self.get("raw").is_none() {
                        return self.raw_op(name, args, *span);
                    }
                }
                let rv = self.eval(recv)?;
                self.method(rv, name, args, *span)
            }
            Expr::Field(recv, name, span) => {
                let rv = self.eval(recv)?;
                self.field(rv, name, *span)
            }
            Expr::Lambda(params, body, span) => {
                // Capture free variables *by value at creation* — scalars copy; collections/structs/
                // tensors are Rc so they stay shared (the language's normal reference semantics).
                let mut used = std::collections::HashSet::new();
                collect_idents_stmts(body, &mut used);
                let mut captured = HashMap::new();
                for n in &used {
                    if params.contains(n) {
                        continue;
                    }
                    if let Some(v) = self.get(n) {
                        captured.insert(n.clone(), v);
                    }
                }
                self.channel.info(
                    *span,
                    format!("closure created · {} param(s) · captures {} var(s) by value (collections stay shared refs)", params.len(), captured.len()),
                );
                Ok(Value::Fn(std::rc::Rc::new(crate::value::FnData { name: None, params: params.clone(), body: body.clone(), captured })))
            }
            Expr::CallValue(callee, args, span) => {
                let cv = self.eval(callee)?;
                let argv = self.eval_args(args)?;
                self.call_fn_value(cv, argv, *span)
            }
        }
    }

    /// Call a function *value* (closure or named-fn-as-value) with already-evaluated arguments (v0.42).
    /// The frame starts from the captured environment, then parameters overwrite.
    fn call_fn_value(&mut self, cv: Value, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let c = match cv {
            Value::Fn(c) => c,
            other => return Err(format!("line {}: {} is not callable", span.line, other.type_name())),
        };
        if argv.len() != c.params.len() {
            let label = c.name.clone().unwrap_or_else(|| "fn".into());
            return Err(format!("line {}: '{}' expects {} arg(s), got {}", span.line, label, c.params.len(), argv.len()));
        }
        let mut frame: HashMap<String, Value> = c.captured.clone();
        for (p, v) in c.params.iter().zip(argv) {
            frame.insert(p.clone(), v);
        }
        let saved = std::mem::replace(&mut self.locals, vec![frame]);
        let saved_fid = self.frame_id;
        self.frame_id = self.next_frame;
        self.next_frame += 1;
        let flow = self.exec_stmts(&c.body);
        self.release_frame_claims(self.frame_id);
        self.frame_id = saved_fid;
        self.locals = saved;
        match flow {
            Ok(Flow::Return(v)) => Ok(v),
            Ok(Flow::Normal) => Ok(Value::Unit),
            Ok(Flow::Break) | Ok(Flow::Continue) => Err(format!("line {}: break/continue outside a loop in a function value", span.line)),
            Err(ref s) if s == PROPAGATE => Ok(self.err_value.take().unwrap_or(Value::Unit)),
            Err(e) => Err(e),
        }
    }

    fn bool_operand(&mut self, e: &Expr, op: &str, span: Span) -> Result<Value, String> {
        let v = self.eval(e)?;
        v.truthy()
            .map(Value::Bool)
            .ok_or_else(|| format!("line {}: '{}' operand must be bool (found {})", span.line, op, v.type_name()))
    }

    fn as_int(&mut self, e: &Expr, span: Span) -> Result<i64, String> {
        match self.eval(e)? {
            Value::Int(n) => Ok(n),
            other => Err(format!("line {}: expected int (found {})", span.line, other.type_name())),
        }
    }

    // ---- indexing ----

    fn index_get(&mut self, recv: Value, idx: Value, span: Span) -> Result<Value, String> {
        // Shared with the VM (runtime, v0.44): negative indexes + range slices, single semantics source.
        crate::runtime::index_read(&recv, &idx, span, &mut self.channel)
    }

    fn index_set(&mut self, recv: Value, idx: Value, val: Value, span: Span) -> Result<(), String> {
        if self.trusted == 0 {
            if let Value::Array(a) = &recv {
                crate::runtime::borrow_check_mut(&self.borrows, a, "index assignment", span, &mut self.channel)?;
            }
        }
        crate::runtime::index_write(&recv, &idx, val, span.line)
    }

    // ---- operators ----

    fn binop(&mut self, op: &BinOp, l: Value, r: Value, span: Span) -> Result<Value, String> {
        use BinOp::*;
        #[cfg(feature = "ai")]
        if matches!(l, Value::Tensor(_)) || matches!(r, Value::Tensor(_)) {
            return self.tensor_op(op, l, r, span);
        }
        match op {
            Add => self.add(l, r, span),
            Sub | Mul | Div => self.arith(op, l, r, span),
            Eq => Ok(Value::Bool(self.equals(&l, &r, span)?)),
            Ne => Ok(Value::Bool(!self.equals(&l, &r, span)?)),
            Lt | Gt | Le | Ge => self.order(op, l, r, span),
        }
    }

    fn add(&mut self, l: Value, r: Value, span: Span) -> Result<Value, String> {
        match (&l, &r) {
            (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (Value::Str(a), Value::Str(b)) => Ok(Value::Str(format!("{}{}", a, b))),
            (Value::Array(a), Value::Array(b)) => {
                let mut v = a.borrow().clone();
                v.extend(b.borrow().iter().cloned());
                Ok(Value::array(v))
            }
            _ => match (to_f(&l), to_f(&r)) {
                (Some(x), Some(y)) => Ok(Value::Float(x + y)),
                _ => Err(format!("line {}: cannot do {} + {}", span.line, l.type_name(), r.type_name())),
            },
        }
    }

    fn arith(&mut self, op: &BinOp, l: Value, r: Value, span: Span) -> Result<Value, String> {
        if let (Value::Int(a), Value::Int(b)) = (&l, &r) {
            let (a, b) = (*a, *b);
            return Ok(match op {
                BinOp::Sub => Value::Int(a - b),
                BinOp::Mul => Value::Int(a * b),
                BinOp::Div => {
                    if b == 0 {
                        self.channel.warn(span, "division by zero — runtime error");
                        return Err(format!("line {}: division by zero", span.line));
                    }
                    Value::Int(a / b)
                }
                _ => unreachable!(),
            });
        }
        match (to_f(&l), to_f(&r)) {
            (Some(x), Some(y)) => Ok(match op {
                BinOp::Sub => Value::Float(x - y),
                BinOp::Mul => Value::Float(x * y),
                BinOp::Div => {
                    if y == 0.0 {
                        self.channel.warn(span, "division by zero (0.0) — runtime error");
                        return Err(format!("line {}: division by zero", span.line));
                    }
                    Value::Float(x / y)
                }
                _ => unreachable!(),
            }),
            _ => Err(format!("line {}: cannot do arithmetic on {} and {}", span.line, l.type_name(), r.type_name())),
        }
    }

    fn equals(&self, l: &Value, r: &Value, span: Span) -> Result<bool, String> {
        if let (Some(x), Some(y)) = (to_f(l), to_f(r)) {
            return Ok(x == y);
        }
        if std::mem::discriminant(l) == std::mem::discriminant(r) {
            return Ok(l == r);
        }
        Err(format!("line {}: {} and {} are not comparable", span.line, l.type_name(), r.type_name()))
    }

    fn order(&self, op: &BinOp, l: Value, r: Value, span: Span) -> Result<Value, String> {
        let ord = value_cmp(&l, &r).map_err(|e| format!("line {}: {}", span.line, e))?;
        let b = match op {
            BinOp::Lt => ord == Ordering::Less,
            BinOp::Gt => ord == Ordering::Greater,
            BinOp::Le => ord != Ordering::Greater,
            BinOp::Ge => ord != Ordering::Less,
            _ => unreachable!(),
        };
        Ok(Value::Bool(b))
    }

    // ---- call / method / field ----

    fn eval_args(&mut self, args: &[Expr]) -> Result<Vec<Value>, String> {
        args.iter().map(|a| self.eval(a)).collect()
    }

    /// Try to JIT-compile the program's functions to native code as one batch (integer subset, `jit`
    /// feature). Batching lets eligible functions call each other — direct and mutual recursion run
    /// entirely as machine code (v0.41). Best-effort: ineligible functions (or a failed batch) silently
    /// fall back to the tree-walker.
    #[cfg(feature = "jit")]
    fn try_jit_batch(&mut self, cands: Vec<(String, Vec<String>, Vec<Stmt>, Span)>) {
        // A function whose name shadows a builtin is unreachable through call() (builtins dispatch
        // first), so it must not be compiled or become a call target.
        let defs: Vec<crate::jit::FnDef> = cands
            .iter()
            .filter(|(n, ..)| !crate::check::CORE_BUILTINS.contains(&n.as_str()) && builtin_module(n).is_none())
            .map(|(n, p, b, _)| (n.clone(), p.clone(), b.clone()))
            .collect();
        let abis = crate::jit::eligible_set(&defs);
        let chosen: Vec<crate::jit::FnDef> = defs.into_iter().filter(|(n, ..)| abis.contains_key(n)).collect();
        if chosen.is_empty() {
            return;
        }
        if self.jit.is_none() {
            self.jit = crate::jit::Jit::new().ok();
        }
        if let Some(j) = self.jit.as_mut() {
            if let Ok(compiled) = j.compile_batch(&chosen, &abis) {
                for (name, c) in compiled {
                    let span = cands.iter().find(|(n, ..)| *n == name).map(|t| t.3).unwrap_or(Span { line: 0, col: 0 });
                    let path = if c.abi() == crate::jit::Abi::I64 { "integer" } else { "float" };
                    self.channel.info(span, format!("JIT compiled '{}' to native code ({} fast path)", name, path));
                    self.jit_fns.insert(name, c);
                }
            }
        }
    }

    fn call(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Value, String> {
        // builtin functions
        if let Some(result) = self.builtin(name, args, span)? {
            return Ok(result);
        }
        // user functions
        let func = match self.funcs.get(name) {
            Some(f) => f.clone(),
            None => {
                // a variable holding a function value is callable by name too (closures, v0.42)
                if let Some(v @ Value::Fn(_)) = self.get(name) {
                    let argv = self.eval_args(args)?;
                    return self.call_fn_value(v, argv, span);
                }
                return Err(format!("line {}: undefined function '{}'", span.line, name));
            }
        };
        if args.len() != func.params.len() {
            return Err(format!(
                "line {}: '{}' expects {} arg(s), got {}",
                span.line, name, func.params.len(), args.len()
            ));
        }
        // Evaluate arguments once (avoids double-evaluating side effects when dispatching to the JIT).
        let argv: Vec<Value> = args.iter().map(|a| self.eval(a)).collect::<Result<_, _>>()?;
        // Native fast path (jit feature): I64-ABI functions take all-int args; F64-ABI take all-float
        // args (int args stay on the tree-walker, which promotes per-op — exact-parity dispatch only).
        #[cfg(feature = "jit")]
        if let Some(compiled) = self.jit_fns.get(name) {
            match compiled.abi() {
                crate::jit::Abi::I64 => {
                    let ints: Option<Vec<i64>> = argv.iter().map(|v| if let Value::Int(n) = v { Some(*n) } else { None }).collect();
                    if let Some(ints) = ints {
                        if let Some(r) = compiled.call(&ints) {
                            self.channel.info(span, format!("native (JIT) call '{}' — {} args", name, ints.len()));
                            return Ok(Value::Int(r));
                        }
                    }
                }
                crate::jit::Abi::F64 => {
                    let floats: Option<Vec<f64>> = argv.iter().map(|v| if let Value::Float(x) = v { Some(*x) } else { None }).collect();
                    if let Some(floats) = floats {
                        if let Some(r) = compiled.call_f64(&floats) {
                            self.channel.info(span, format!("native (JIT) call '{}' — {} float args", name, floats.len()));
                            return Ok(Value::Float(r));
                        }
                    }
                }
            }
        }
        let mut frame = HashMap::new();
        for (p, v) in func.params.iter().zip(argv) {
            frame.insert(p.clone(), v);
        }
        let saved = std::mem::replace(&mut self.locals, vec![frame]);
        let saved_fid = self.frame_id;
        self.frame_id = self.next_frame;
        self.next_frame += 1;
        let flow = self.exec_stmts(&func.body);
        self.release_frame_claims(self.frame_id); // borrows made in this frame end with it (v0.48)
        self.frame_id = saved_fid;
        self.locals = saved;
        match flow {
            Ok(Flow::Return(v)) => Ok(v),
            Ok(Flow::Normal) => Ok(Value::Unit),
            Ok(Flow::Break) | Ok(Flow::Continue) => {
                Err(format!("line {}: break/continue outside a loop in function '{}'", span.line, name))
            }
            // Turn the error value propagated by `?` back into this function's return value.
            Err(ref s) if s == PROPAGATE => Ok(self.err_value.take().unwrap_or(Value::Unit)),
            Err(e) => Err(e),
        }
    }

    /// Builtin functions. Some(value) if matched, otherwise None (→ try user function).
    fn builtin(&mut self, name: &str, args: &[Expr], span: Span) -> Result<Option<Value>, String> {
        let line = span.line;
        // std module gating — error if that module hasn't been imported.
        if let Some(m) = builtin_module(name) {
            // `ai` builtins also need the `ai` build feature compiled in (v0.17). When it's off the
            // tensor arms below don't exist, so report it clearly instead of "undefined function".
            if m == "ai" && !cfg!(feature = "ai") {
                return Err(format!("line {}: '{}' requires the `ai` build feature (rebuild with --features ai)", line, name));
            }
            if !self.std_modules.contains(m) {
                return Err(format!("line {}: '{}' requires `import \"std/{}\"`", line, name, m));
            }
        }
        let result = match name {
            "print" => {
                let argv = self.eval_args(args)?;
                // Using a gpu tensor on the host needs a D2H transfer — illuminate (§4.3).
                #[cfg(feature = "ai")]
                for v in &argv {
                    if let Value::Tensor(t) = v {
                        let (dev, bytes) = {
                            let tb = t.borrow();
                            (tb.device, tb.bytes())
                        };
                        if dev == Device::Gpu {
                            self.channel.info(span, format!("D2H: {} B → host (for output)", bytes));
                        }
                    }
                }
                let parts: Vec<String> = argv.iter().map(|v| v.to_string()).collect();
                println!("{}", parts.join(" "));
                Value::Unit
            }
            // file I/O (std/fs, v0.43) — one implementation in runtime (shared with the VM).
            "read_file" | "read_lines" | "write_file" | "append_file" | "remove_file" | "file_exists" => {
                let argv = self.eval_args(args)?;
                match crate::runtime::value_builtin(name, &argv, span, &mut self.channel) {
                    Some(r) => r?,
                    None => return Ok(None),
                }
            }
            "heap" => {
                self.arity(args, 0, name, line)?;
                self.channel.info(span, "heap (priority queue, mutable)");
                Value::empty_heap()
            }
            "set" => {
                self.arity(args, 0, name, line)?;
                self.channel.info(span, "heap set (mutable)");
                Value::Set(std::rc::Rc::new(std::cell::RefCell::new(std::collections::BTreeSet::new())))
            }
            "strbuf" => {
                self.arity(args, 0, name, line)?;
                self.channel.info(span, "string builder (amortized O(1) append — avoids `s = s + c` O(n²))");
                Value::StrBuf(std::rc::Rc::new(std::cell::RefCell::new(String::new())))
            }
            "len" => {
                let v = self.one(args, name, line)?;
                Value::Int(length(&v).ok_or_else(|| format!("line {}: len() cannot be applied to {}", line, v.type_name()))?)
            }
            "int" => {
                let argv = self.eval_args(args)?;
                to_int(&argv, line)?
            }
            "float" => {
                let v = self.one(args, name, line)?;
                match v {
                    Value::Int(n) => Value::Float(n as f64),
                    Value::Float(x) => Value::Float(x),
                    Value::Str(s) => Value::Float(
                        s.trim().parse().map_err(|_| format!("line {}: failed to parse float '{}'", line, s))?,
                    ),
                    other => return Err(format!("line {}: float() cannot be applied to {}", line, other.type_name())),
                }
            }
            "str" => {
                let v = self.one(args, name, line)?;
                Value::Str(v.to_string())
            }
            "abs" => match self.one(args, name, line)? {
                Value::Int(n) => Value::Int(n.abs()),
                Value::Float(x) => Value::Float(x.abs()),
                other => return Err(format!("line {}: abs() takes numbers only (found {})", line, other.type_name())),
            },
            "sqrt" => Value::Float(self.one_num(args, name, line)?.sqrt()),
            "floor" => Value::Int(self.one_num(args, name, line)?.floor() as i64),
            "ceil" => Value::Int(self.one_num(args, name, line)?.ceil() as i64),
            "pow" => {
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                pow(&argv[0], &argv[1], line)?
            }
            "min" | "max" => {
                let argv = self.eval_args(args)?;
                if argv.is_empty() {
                    return Err(format!("line {}: {}() needs at least 1 arg", line, name));
                }
                let want_less = name == "min";
                let mut best = argv[0].clone();
                for v in &argv[1..] {
                    let ord = value_cmp(v, &best).map_err(|e| format!("line {}: {}", line, e))?;
                    if (want_less && ord == Ordering::Less) || (!want_less && ord == Ordering::Greater) {
                        best = v.clone();
                    }
                }
                best
            }
            "hex" => Value::Str(radix_str(self.one_int(args, name, line)?, 16, "0x")),
            "bin" => Value::Str(radix_str(self.one_int(args, name, line)?, 2, "0b")),
            "ord" => {
                let v = self.one(args, name, line)?;
                match v {
                    Value::Str(s) if s.chars().count() == 1 => Value::Int(s.chars().next().unwrap() as i64),
                    _ => return Err(format!("line {}: ord() takes a single-character string only", line)),
                }
            }
            "chr" => {
                let n = self.one_int(args, name, line)?;
                let c = u32::try_from(n)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| format!("line {}: chr({}) invalid code point", line, n))?;
                Value::Str(c.to_string())
            }
            #[cfg(feature = "ai")]
            "tensor" => {
                let v = self.one(args, name, line)?;
                let td = build_tensor(&v).map_err(|e| format!("line {}: tensor() — {}", line, e))?;
                self.illuminate_tensor(span, &td, "created");
                Value::Tensor(rc_tensor(td))
            }
            #[cfg(feature = "ai")]
            "param" => {
                // Trainable leaf tensor (requires_grad). The starting point for backprop.
                let v = self.one(args, name, line)?;
                let mut td = build_tensor(&v).map_err(|e| format!("line {}: param() — {}", line, e))?;
                td.requires_grad = true;
                self.illuminate_tensor(span, &td, "param created (grad tracked)");
                Value::Tensor(rc_tensor(td))
            }
            #[cfg(feature = "ai")]
            "zeros" | "ones" => {
                let v = self.one(args, name, line)?;
                let shape = shape_from(&v).map_err(|e| format!("line {}: {}() — {}", line, name, e))?;
                let total: usize = shape.iter().product();
                let fill = if name == "ones" { 1.0 } else { 0.0 };
                let td = TensorData::new(shape, vec![fill; total]);
                self.illuminate_tensor(span, &td, "created");
                Value::Tensor(rc_tensor(td))
            }
            #[cfg(feature = "ai")]
            "matmul" => {
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                self.matmul(&argv[0], &argv[1], span)?
            }
            #[cfg(feature = "ai")]
            "conv2d" => {
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                self.conv2d(&argv[0], &argv[1], span)?
            }
            #[cfg(feature = "ai")]
            "maxpool2d" => {
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                self.maxpool2d(&argv[0], &argv[1], span)?
            }
            #[cfg(feature = "ai")]
            "grad_step" => {
                // In-place SGD: w.data -= lr * w.grad, then reset grad (PyTorch step+zero_grad).
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                let w = match &argv[0] {
                    Value::Tensor(t) => t.clone(),
                    other => return Err(format!("line {}: grad_step(tensor, lr) — first arg is {}", line, other.type_name())),
                };
                let lr = to_f(&argv[1]).ok_or_else(|| format!("line {}: lr must be a number", line))? as f32;
                let mut wb = w.borrow_mut();
                let grad = wb.grad.clone().ok_or_else(|| format!("line {}: no grad — run backward first", line))?;
                for i in 0..wb.data.len() {
                    wb.data[i] -= lr * grad[i];
                }
                wb.grad = None; // reset for the next step
                #[cfg(feature = "gpu")]
                {
                    wb.gpu_buf = None; // host data changed — the cached device buffer is stale
                }
                Value::Unit
            }
            #[cfg(feature = "ai")]
            "adam_step" => {
                // In-place Adam: bias-corrected moment estimates (β1=0.9, β2=0.999, ε=1e-8). State
                // (m, v, t) lives in the tensor, so it persists across steps for the same parameter.
                let argv = self.eval_args(args)?;
                self.arity_v(&argv, 2, name, line)?;
                let w = match &argv[0] {
                    Value::Tensor(t) => t.clone(),
                    other => return Err(format!("line {}: adam_step(tensor, lr) — first arg is {}", line, other.type_name())),
                };
                let lr = to_f(&argv[1]).ok_or_else(|| format!("line {}: lr must be a number", line))? as f32;
                let (b1, b2, eps) = (0.9f32, 0.999f32, 1e-8f32);
                let mut wb = w.borrow_mut();
                let grad = wb.grad.clone().ok_or_else(|| format!("line {}: no grad — run backward first", line))?;
                let n = wb.data.len();
                if wb.adam_m.is_none() {
                    wb.adam_m = Some(vec![0.0; n]);
                    wb.adam_v = Some(vec![0.0; n]);
                }
                wb.adam_t += 1;
                let t = wb.adam_t;
                let bc1 = 1.0 - b1.powi(t as i32); // bias correction
                let bc2 = 1.0 - b2.powi(t as i32);
                let mut m = wb.adam_m.take().unwrap();
                let mut v = wb.adam_v.take().unwrap();
                for i in 0..n {
                    m[i] = b1 * m[i] + (1.0 - b1) * grad[i];
                    v[i] = b2 * v[i] + (1.0 - b2) * grad[i] * grad[i];
                    let mhat = m[i] / bc1;
                    let vhat = v[i] / bc2;
                    wb.data[i] -= lr * mhat / (vhat.sqrt() + eps);
                }
                wb.adam_m = Some(m);
                wb.adam_v = Some(v);
                wb.grad = None; // reset for the next step
                #[cfg(feature = "gpu")]
                {
                    wb.gpu_buf = None; // host data changed — the cached device buffer is stale
                }
                self.channel.info(span, format!("adam step {} · {} params (moment state in-tensor)", t, n));
                Value::Unit
            }
            #[cfg(feature = "ai")]
            "relu" => self.unary_tensor(args, name, span, GradOp::Relu, |x| x.max(0.0))?,
            #[cfg(feature = "ai")]
            "sigmoid" => self.unary_tensor(args, name, span, GradOp::Sigmoid, |x| 1.0 / (1.0 + (-x).exp()))?,
            #[cfg(feature = "ai")]
            "tanh" => self.unary_tensor(args, name, span, GradOp::Tanh, |x| x.tanh())?,
            #[cfg(feature = "ai")]
            "exp" => self.unary_tensor(args, name, span, GradOp::Exp, |x| x.exp())?,
            #[cfg(feature = "ai")]
            "log" => self.unary_tensor(args, name, span, GradOp::Log, |x| x.ln())?,
            #[cfg(feature = "ai")]
            "softmax" => {
                // Row-wise softmax (2D=(batch,classes), 1D=(1,n)). Numerically stable (subtract max).
                let t = match self.one(args, name, line)? {
                    Value::Tensor(t) => t,
                    other => return Err(format!("line {}: softmax(tensor) — {}", line, other.type_name())),
                };
                let (data, shape, dev) = {
                    let b = t.borrow();
                    (b.data.clone(), b.shape.clone(), b.device)
                };
                let (rows, cols) = rows_cols(&shape);
                let out_data = softmax_rows(&data, rows, cols);
                let mut out = TensorData::new(shape, out_data);
                out.device = dev;
                self.illuminate_tensor(span, &out, "softmax");
                self.record_tape(span, &mut out, Some(GradNode { op: GradOp::Softmax, inputs: vec![t] }));
                Value::Tensor(rc_tensor(out))
            }
            #[cfg(feature = "ai")]
            "transpose" => {
                let t = match self.one(args, name, line)? {
                    Value::Tensor(t) => t,
                    other => return Err(format!("line {}: transpose(tensor) — {}", line, other.type_name())),
                };
                let (data, shape, dev) = {
                    let b = t.borrow();
                    if b.shape.len() != 2 {
                        return Err(format!("line {}: transpose works on 2D only ({}D)", line, b.shape.len()));
                    }
                    let (m, n) = (b.shape[0], b.shape[1]);
                    let mut d = vec![0f32; m * n];
                    for i in 0..m {
                        for j in 0..n {
                            d[j * m + i] = b.data[i * n + j];
                        }
                    }
                    (d, vec![n, m], b.device)
                };
                let mut out = TensorData::new(shape, data);
                out.device = dev;
                self.illuminate_tensor(span, &out, "transpose");
                self.record_tape(span, &mut out, Some(GradNode { op: GradOp::Transpose, inputs: vec![t] }));
                Value::Tensor(rc_tensor(out))
            }
            "err" => {
                let v = self.one(args, name, line)?;
                Value::Err(Box::new(v))
            }
            "is_err" => {
                let v = self.one(args, name, line)?;
                Value::Bool(matches!(v, Value::Err(_)))
            }
            "err_msg" => match self.one(args, name, line)? {
                Value::Err(p) => *p,
                other => other, // if not an error, return the value as-is
            },
            "assert" => {
                let argv = self.eval_args(args)?;
                if argv.is_empty() || argv.len() > 2 {
                    return Err(format!("line {}: assert(cond[, msg])", line));
                }
                let ok = argv[0].truthy().ok_or_else(|| format!("line {}: assert condition must be bool", line))?;
                if !ok {
                    let msg = argv.get(1).map(|v| v.to_string()).unwrap_or_else(|| "assertion failed".into());
                    self.channel.warn(span, format!("assertion failed: {}", msg));
                    return Err(format!("line {}: assertion failed: {}", line, msg));
                }
                Value::Unit
            }
            _ => return Ok(None),
        };
        Ok(Some(result))
    }

    fn method(&mut self, recv: Value, name: &str, args: &[Expr], span: Span) -> Result<Value, String> {
        let argv = self.eval_args(args)?;
        match &recv {
            Value::Array(a) => self.array_method(&a.clone(), name, argv, span),
            Value::Str(s) => self.str_method(&s.clone(), name, argv, span),
            Value::Map(m) => self.map_method(&m.clone(), name, argv, span),
            Value::Heap(h) => self.heap_method(&h.clone(), name, argv, span),
            Value::Set(s) => self.set_method(&s.clone(), name, argv, span),
            Value::StrBuf(b) => self.strbuf_method(&b.clone(), name, argv, span),
            #[cfg(feature = "ai")]
            Value::Tensor(t) => self.tensor_method(&t.clone(), name, argv, span),
            Value::Struct { name: sname, .. } => self.struct_method(recv.clone(), &sname.clone(), name, argv, span),
            other => Err(format!("line {}: {} has no method '{}'", span.line, other.type_name(), name)),
        }
    }

    /// Calls a user-defined struct method. Binds the receiver to the method's first parameter `self`
    /// (reference semantics, so self.field = ... is visible to the caller). The remaining parameters
    /// follow argv order. (v0.16 impl)
    fn struct_method(&mut self, recv: Value, sname: &str, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let func = self
            .methods
            .get(sname)
            .and_then(|t| t.get(name))
            .cloned()
            .ok_or_else(|| format!("line {}: {} has no method '{}'", span.line, sname, name))?;
        if func.params.is_empty() {
            return Err(format!("line {}: method '{}.{}' has no self parameter", span.line, sname, name));
        }
        let want = func.params.len() - 1; // excluding self
        if argv.len() != want {
            return Err(format!(
                "line {}: method '{}.{}' expects {} arg(s), got {}",
                span.line, sname, name, want, argv.len()
            ));
        }
        let mut frame = HashMap::new();
        frame.insert(func.params[0].clone(), recv); // self
        for (p, v) in func.params[1..].iter().zip(argv) {
            frame.insert(p.clone(), v);
        }
        let saved = std::mem::replace(&mut self.locals, vec![frame]);
        let flow = self.exec_stmts(&func.body);
        self.locals = saved;
        match flow {
            Ok(Flow::Return(v)) => Ok(v),
            Ok(Flow::Normal) => Ok(Value::Unit),
            Ok(Flow::Break) | Ok(Flow::Continue) => {
                Err(format!("line {}: break/continue outside a loop in method '{}.{}'", span.line, sname, name))
            }
            Err(ref s) if s == PROPAGATE => Ok(self.err_value.take().unwrap_or(Value::Unit)),
            Err(e) => Err(e),
        }
    }

    fn array_method(&mut self, a: &ArrayRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        // Borrow gradient (§3.3): a mutating method needs exclusive access — reject if shared-borrowed.
        // (`@trust` turns the check off for its statement, v0.48.)
        if self.trusted == 0 && crate::runtime::MUT_ARRAY_METHODS.contains(&name) {
            crate::runtime::borrow_check_mut(&self.borrows, a, name, span, &mut self.channel)?;
        }
        match name {
            "push" => {
                check_n(&argv, 1, name, line)?;
                a.borrow_mut().push(argv[0].clone());
                Ok(Value::Unit)
            }
            "push_front" => {
                check_n(&argv, 1, name, line)?;
                a.borrow_mut().insert(0, argv[0].clone());
                Ok(Value::Unit)
            }
            "pop" => {
                check_n(&argv, 0, name, line)?;
                a.borrow_mut().pop().ok_or_else(|| {
                    self.channel.warn(span, "pop from empty array");
                    format!("line {}: pop from empty array", line)
                })
            }
            "pop_front" => {
                check_n(&argv, 0, name, line)?;
                let mut b = a.borrow_mut();
                if b.is_empty() {
                    self.channel.warn(span, "pop_front from empty array");
                    return Err(format!("line {}: pop_front from empty array", line));
                }
                Ok(b.remove(0))
            }
            "insert" => {
                check_n(&argv, 2, name, line)?;
                let i = as_index(&argv[0], a.borrow().len() + 1, line)?;
                a.borrow_mut().insert(i, argv[1].clone());
                Ok(Value::Unit)
            }
            "remove" => {
                check_n(&argv, 1, name, line)?;
                let i = as_index(&argv[0], a.borrow().len(), line)?;
                Ok(a.borrow_mut().remove(i))
            }
            "contains" => {
                check_n(&argv, 1, name, line)?;
                let found = a.borrow().iter().any(|e| e == &argv[0]);
                Ok(Value::Bool(found))
            }
            // Higher-order methods (v0.42 closures) — iterate a snapshot, build a *new* array.
            "map" | "filter" => {
                check_n(&argv, 1, name, line)?;
                let f = argv.into_iter().next().unwrap();
                if !matches!(f, Value::Fn(_)) {
                    return Err(format!("line {}: .{}(fn) takes a function value (found {})", line, name, f.type_name()));
                }
                let items = a.borrow().clone();
                let mut out = Vec::new();
                for it in items {
                    let r = self.call_fn_value(f.clone(), vec![it.clone()], span)?;
                    if name == "map" {
                        out.push(r);
                    } else {
                        match r {
                            Value::Bool(true) => out.push(it),
                            Value::Bool(false) => {}
                            other => return Err(format!("line {}: filter fn must return bool (found {})", line, other.type_name())),
                        }
                    }
                }
                self.channel.info(span, format!("{} → new heap vector · {} elems", name, out.len()));
                Ok(Value::array(out))
            }
            "reverse" => {
                check_n(&argv, 0, name, line)?;
                a.borrow_mut().reverse();
                Ok(Value::Unit)
            }
            "sort" => {
                check_n(&argv, 0, name, line)?;
                let mut b = a.borrow_mut();
                for w in b.windows(2) {
                    value_cmp(&w[0], &w[1]).map_err(|e| format!("line {}: sort — {}", line, e))?;
                }
                b.sort_by(|x, y| value_cmp(x, y).unwrap_or(Ordering::Equal));
                Ok(Value::Unit)
            }
            "clear" => {
                check_n(&argv, 0, name, line)?;
                a.borrow_mut().clear();
                Ok(Value::Unit)
            }
            "sum" => {
                check_n(&argv, 0, name, line)?;
                sum_array(&a.borrow(), line)
            }
            "join" => {
                check_n(&argv, 1, name, line)?;
                let sep = match &argv[0] {
                    Value::Str(s) => s.clone(),
                    other => return Err(format!("line {}: join(sep) — sep must be str (found {})", line, other.type_name())),
                };
                let parts: Vec<String> = a.borrow().iter().map(|v| v.to_string()).collect();
                Ok(Value::Str(parts.join(&sep)))
            }
            _ => Err(format!("line {}: array has no method '{}'", line, name)),
        }
    }

    fn heap_method(&mut self, h: &HeapRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        match name {
            "push" => {
                check_n(&argv, 1, name, line)?;
                heap_push(&mut h.borrow_mut(), argv[0].clone()).map_err(|e| format!("line {}: {}", line, e))?;
                Ok(Value::Unit)
            }
            "pop" => {
                check_n(&argv, 0, name, line)?;
                match heap_pop(&mut h.borrow_mut()).map_err(|e| format!("line {}: {}", line, e))? {
                    Some(v) => Ok(v),
                    None => {
                        self.channel.warn(span, "pop from empty heap");
                        Err(format!("line {}: pop from empty heap", line))
                    }
                }
            }
            "peek" => {
                check_n(&argv, 0, name, line)?;
                h.borrow().first().cloned().ok_or_else(|| format!("line {}: peek on empty heap", line))
            }
            _ => Err(format!("line {}: heap has no method '{}'", line, name)),
        }
    }

    fn set_method(&mut self, s: &SetRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        let key = |v: &Value| v.as_key().ok_or_else(|| format!("line {}: set elements must be int, str, or bool", line));
        match name {
            "add" => {
                check_n(&argv, 1, name, line)?;
                s.borrow_mut().insert(key(&argv[0])?);
                Ok(Value::Unit)
            }
            "contains" => {
                check_n(&argv, 1, name, line)?;
                Ok(Value::Bool(s.borrow().contains(&key(&argv[0])?)))
            }
            "remove" => {
                check_n(&argv, 1, name, line)?;
                Ok(Value::Bool(s.borrow_mut().remove(&key(&argv[0])?)))
            }
            "items" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::array(s.borrow().iter().map(|k| k.to_value()).collect()))
            }
            _ => Err(format!("line {}: set has no method '{}'", line, name)),
        }
    }

    fn map_method(&mut self, m: &MapRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        match name {
            "get" => {
                if argv.is_empty() || argv.len() > 2 {
                    return Err(format!("line {}: get(key[, default])", line));
                }
                let key = argv[0].as_key().ok_or_else(|| format!("line {}: invalid map key", line))?;
                match m.borrow().get(&key).cloned() {
                    Some(v) => Ok(v),
                    None => Ok(argv.get(1).cloned().unwrap_or(Value::Unit)),
                }
            }
            "contains" => {
                check_n(&argv, 1, name, line)?;
                let key = argv[0].as_key().ok_or_else(|| format!("line {}: invalid map key", line))?;
                Ok(Value::Bool(m.borrow().contains_key(&key)))
            }
            "remove" => {
                check_n(&argv, 1, name, line)?;
                let key = argv[0].as_key().ok_or_else(|| format!("line {}: invalid map key", line))?;
                Ok(m.borrow_mut().remove(&key).unwrap_or(Value::Unit))
            }
            "keys" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::array(m.borrow().keys().map(|k| k.to_value()).collect()))
            }
            "values" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::array(m.borrow().values().cloned().collect()))
            }
            _ => Err(format!("line {}: map has no method '{}'", line, name)),
        }
    }

    fn str_method(&mut self, s: &str, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        let str_arg = |v: &Value| -> Result<String, String> {
            match v {
                Value::Str(s) => Ok(s.clone()),
                other => Err(format!("line {}: expected str argument (found {})", line, other.type_name())),
            }
        };
        match name {
            "upper" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::Str(s.to_uppercase()))
            }
            "lower" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::Str(s.to_lowercase()))
            }
            "trim" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::Str(s.trim().to_string()))
            }
            "split" => {
                check_n(&argv, 1, name, line)?;
                let sep = str_arg(&argv[0])?;
                let parts: Vec<Value> = if sep.is_empty() {
                    s.chars().map(|c| Value::Str(c.to_string())).collect()
                } else {
                    s.split(&sep as &str).map(|p| Value::Str(p.to_string())).collect()
                };
                Ok(Value::array(parts))
            }
            "chars" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::array(s.chars().map(|c| Value::Str(c.to_string())).collect()))
            }
            "contains" => {
                check_n(&argv, 1, name, line)?;
                Ok(Value::Bool(s.contains(&str_arg(&argv[0])?)))
            }
            "starts_with" => {
                check_n(&argv, 1, name, line)?;
                Ok(Value::Bool(s.starts_with(&str_arg(&argv[0])?)))
            }
            "ends_with" => {
                check_n(&argv, 1, name, line)?;
                Ok(Value::Bool(s.ends_with(&str_arg(&argv[0])?)))
            }
            "replace" => {
                check_n(&argv, 2, name, line)?;
                Ok(Value::Str(s.replace(&str_arg(&argv[0])?, &str_arg(&argv[1])?)))
            }
            "find" => {
                check_n(&argv, 1, name, line)?;
                let sub = str_arg(&argv[0])?;
                let idx = s.find(&sub).map(|byte| s[..byte].chars().count() as i64).unwrap_or(-1);
                Ok(Value::Int(idx))
            }
            _ => Err(format!("line {}: str has no method '{}'", line, name)),
        }
    }

    /// Mutable string builder methods (v0.16). push(str|strbuf), str(), clear(). append is amortized O(1).
    fn strbuf_method(&mut self, b: &StrBufRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        match name {
            "push" => {
                check_n(&argv, 1, name, line)?;
                match &argv[0] {
                    Value::Str(s) => b.borrow_mut().push_str(s),
                    Value::StrBuf(o) => {
                        let s = o.borrow().clone();
                        b.borrow_mut().push_str(&s);
                    }
                    other => return Err(format!("line {}: push(x) — x must be str/strbuf (found {})", line, other.type_name())),
                }
                Ok(Value::Unit)
            }
            "str" => {
                check_n(&argv, 0, name, line)?;
                Ok(Value::Str(b.borrow().clone()))
            }
            "clear" => {
                check_n(&argv, 0, name, line)?;
                b.borrow_mut().clear();
                Ok(Value::Unit)
            }
            _ => Err(format!("line {}: strbuf has no method '{}'", line, name)),
        }
    }

    fn struct_lit(&mut self, name: &str, fields: &[(String, Expr)], span: Span) -> Result<Value, String> {
        let decl = self
            .structs
            .get(name)
            .ok_or_else(|| format!("line {}: undefined struct '{}'", span.line, name))?
            .clone();
        let mut map = std::collections::BTreeMap::new();
        for (fname, fexpr) in fields {
            if !decl.contains(fname) {
                return Err(format!("line {}: {} has no field '{}'", span.line, name, fname));
            }
            let v = self.eval(fexpr)?;
            map.insert(fname.clone(), v);
        }
        for d in &decl {
            if !map.contains_key(d) {
                return Err(format!("line {}: struct {} missing field '{}'", span.line, name, d));
            }
        }
        self.channel.info(span, format!("heap struct {} (mutable, shared ref)", name));
        Ok(Value::Struct {
            name: name.to_string(),
            fields: std::rc::Rc::new(std::cell::RefCell::new(map)),
        })
    }

    fn enum_lit(&mut self, ename: &str, variant: &str, args: &[Expr], span: Span) -> Result<Value, String> {
        let arity = *self
            .enums
            .get(ename)
            .ok_or_else(|| format!("line {}: undefined enum '{}'", span.line, ename))?
            .get(variant)
            .ok_or_else(|| format!("line {}: {} has no variant '{}'", span.line, ename, variant))?;
        if args.len() != arity {
            return Err(format!(
                "line {}: {}::{} expects {} arg(s), got {}",
                span.line, ename, variant, arity, args.len()
            ));
        }
        let payload = self.eval_args(args)?;
        Ok(Value::Enum {
            name: ename.to_string(),
            variant: variant.to_string(),
            payload,
        })
    }

}

// ---- tensors (AI — the soul of wide) — gated by the `ai` build feature (v0.17) ----
#[cfg(feature = "ai")]
impl Interp {
    /// Illuminate tensor cost (T2: make every cost visible). dtype, shape, bytes, residency.
    fn illuminate_tensor(&mut self, span: Span, t: &TensorData, what: &str) {
        self.channel.info(
            span,
            format!("tensor {} [{}] {} · {} B · resides on {}", t.dtype, dims_str(&t.shape), what, t.bytes(), t.device),
        );
    }

    /// Decide the result residency from the two operands + illuminate transfers (§4.3).
    /// Both gpu → stay (0 transfer, the key to eliminating redundant transfers). Mixed → auto H2D on the host side.
    fn combine_device(&mut self, a: &TensorData, b: &TensorData, span: Span) -> Device {
        match (a.device, b.device) {
            (Device::Host, Device::Host) => Device::Host,
            (Device::Gpu, Device::Gpu) => {
                self.channel.info(span, "both inputs reside on device — no transfer (redundant transfer eliminated)");
                Device::Gpu
            }
            (Device::Gpu, Device::Host) => {
                self.channel.info(span, format!("auto H2D: {} B → device (host input)", b.bytes()));
                Device::Gpu
            }
            (Device::Host, Device::Gpu) => {
                self.channel.info(span, format!("auto H2D: {} B → device (host input)", a.bytes()));
                Device::Gpu
            }
        }
    }

    fn tensor_op(&mut self, op: &BinOp, l: Value, r: Value, span: Span) -> Result<Value, String> {
        if !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
            return Err(format!("line {}: tensors support arithmetic (+ - * /) only", span.line));
        }
        let apply = |x: f32, y: f32| match op {
            BinOp::Add => x + y,
            BinOp::Sub => x - y,
            BinOp::Mul => x * y,
            BinOp::Div => x / y,
            _ => 0.0,
        };
        let opcode = match op {
            BinOp::Add => 0u32,
            BinOp::Sub => 1,
            BinOp::Mul => 2,
            _ => 3,
        };
        let (mut result, gnode): (TensorData, Option<GradNode>) = match (&l, &r) {
            (Value::Tensor(arc), Value::Tensor(brc)) => {
                let (rsh, device, same_shape) = {
                    let (a, b) = (arc.borrow(), brc.borrow());
                    let rsh = match broadcast_shapes(&a.shape, &b.shape) {
                        Some(s) => s,
                        None => {
                            self.channel.warn(span, format!("cannot broadcast: [{}] vs [{}]", dims_str(&a.shape), dims_str(&b.shape)));
                            return Err(format!("line {}: cannot broadcast [{}] vs [{}]", span.line, dims_str(&a.shape), dims_str(&b.shape)));
                        }
                    };
                    let device = self.combine_device(&a, &b, span);
                    let same = a.shape == b.shape;
                    (rsh, device, same)
                };
                // Real GPU elementwise (v0.51): same-shape gpu-resident operands run the WGSL shader
                // (broadcast shapes fall back to the CPU path — honest gating).
                let td = match self.gpu_elementwise_tt(arc, brc, opcode, &rsh, device, same_shape, span) {
                    Some(td) => td,
                    None => {
                        let (a, b) = (arc.borrow(), brc.borrow());
                        let data = bcast_binary(&a.data, &a.shape, &b.data, &b.shape, &rsh, apply);
                        let mut td = TensorData::new(rsh, data);
                        td.device = device;
                        td
                    }
                };
                let gop = match op {
                    BinOp::Add => Some(GradOp::Add),
                    BinOp::Sub => Some(GradOp::Sub),
                    BinOp::Mul => Some(GradOp::MulElem),
                    _ => None, // div is nonlinear — grad not tracked (later)
                };
                let gnode = gop.map(|o| GradNode { op: o, inputs: vec![arc.clone(), brc.clone()] });
                (td, gnode)
            }
            (Value::Tensor(arc), other) => {
                let s = to_f(other).ok_or_else(|| format!("line {}: cannot operate on tensor and {}", span.line, other.type_name()))? as f32;
                let td = match self.gpu_elementwise_ts(arc, opcode, s, false, span) {
                    Some(td) => td,
                    None => {
                        let a = arc.borrow();
                        let data = a.data.iter().map(|x| apply(*x, s)).collect();
                        let mut td = TensorData::new(a.shape.clone(), data); // scalar broadcast
                        td.device = a.device;
                        td
                    }
                };
                // tensor∘scalar: grad always reduces to ScalarMul(factor).
                let factor = match op {
                    BinOp::Add | BinOp::Sub => 1.0,
                    BinOp::Mul => s,
                    BinOp::Div => 1.0 / s,
                    _ => 1.0,
                };
                (td, Some(GradNode { op: GradOp::ScalarMul(factor), inputs: vec![arc.clone()] }))
            }
            (other, Value::Tensor(brc)) => {
                let s = to_f(other).ok_or_else(|| format!("line {}: cannot operate on {} and tensor", span.line, other.type_name()))? as f32;
                let td = match self.gpu_elementwise_ts(brc, opcode, s, true, span) {
                    Some(td) => td,
                    None => {
                        let b = brc.borrow();
                        let data = b.data.iter().map(|y| apply(s, *y)).collect();
                        let mut td = TensorData::new(b.shape.clone(), data);
                        td.device = b.device;
                        td
                    }
                };
                let (factor, track) = match op {
                    BinOp::Add => (1.0, true),
                    BinOp::Sub => (-1.0, true),
                    BinOp::Mul => (s, true),
                    _ => (0.0, false), // s/t nonlinear — not tracked
                };
                let gnode = track.then(|| GradNode { op: GradOp::ScalarMul(factor), inputs: vec![brc.clone()] });
                (td, gnode)
            }
            _ => unreachable!(),
        };
        self.record_tape(span, &mut result, gnode);
        self.illuminate_tensor(span, &result, "elementwise");
        Ok(Value::Tensor(rc_tensor(result)))
    }

    /// Records grad_fn on the result — only when some input is grad-tracked (A2: tape illumination).
    fn record_tape(&mut self, span: Span, result: &mut TensorData, gnode: Option<GradNode>) {
        if let Some(gnode) = gnode {
            let track = gnode.inputs.iter().any(|r| {
                let b = r.borrow();
                b.requires_grad || b.grad_fn.is_some()
            });
            if track {
                result.requires_grad = true;
                result.grad_fn = Some(Rc::new(gnode));
                self.channel.info(span, format!("autodiff graph recorded · stored {} B (activation memory)", result.bytes()));
            }
        }
    }

    /// Unary elementwise tensor op (relu/sigmoid/tanh/exp/log) + tape recording.
    fn unary_tensor(&mut self, args: &[Expr], name: &str, span: Span, op: GradOp, f: impl Fn(f32) -> f32) -> Result<Value, String> {
        let t = match self.one(args, name, span.line)? {
            Value::Tensor(t) => t,
            other => return Err(format!("line {}: {}(tensor) — {}", span.line, name, other.type_name())),
        };
        let (data, shape, dev) = {
            let b = t.borrow();
            (b.data.iter().map(|x| f(*x)).collect::<Vec<f32>>(), b.shape.clone(), b.device)
        };
        let mut out = TensorData::new(shape, data);
        out.device = dev;
        self.illuminate_tensor(span, &out, name);
        self.record_tape(span, &mut out, Some(GradNode { op, inputs: vec![t] }));
        Ok(Value::Tensor(rc_tensor(out)))
    }

    fn matmul(&mut self, a: &Value, b: &Value, span: Span) -> Result<Value, String> {
        let (arc, brc) = match (a, b) {
            (Value::Tensor(a), Value::Tensor(b)) => (a.clone(), b.clone()),
            _ => return Err(format!("line {}: matmul takes two tensors only", span.line)),
        };
        let at = arc.borrow();
        let bt = brc.borrow();
        if at.shape.len() != 2 || bt.shape.len() != 2 {
            return Err(format!("line {}: matmul takes 2D tensors only (got {}D·{}D)", span.line, at.shape.len(), bt.shape.len()));
        }
        let (m, k) = (at.shape[0], at.shape[1]);
        let (k2, n) = (bt.shape[0], bt.shape[1]);
        if k != k2 {
            self.channel.warn(span, format!("matmul dimension mismatch: [{},{}]·[{},{}] — inner {}≠{}", m, k, k2, n, k, k2));
            return Err(format!("line {}: matmul dimension mismatch [{},{}]·[{},{}], inner {}≠{}", span.line, m, k, k2, n, k, k2));
        }
        let device = self.combine_device(&at, &bt, span);
        let a_data = at.data.clone();
        let b_data = bt.data.clone();
        drop(at);
        drop(bt);
        // Real GPU compute (`gpu` feature, v0.50): gpu-resident operands run a WGSL shader. Cached
        // buffers mean a chain re-uploads nothing (§4.3); the result's readback (D2H) is illuminated
        // honestly (lazy downloads are a later refinement). No adapter → CPU fallback with an INFO.
        #[cfg(feature = "gpu")]
        if device == Device::Gpu {
            if let Some(g) = crate::gpu::ctx() {
                let abuf = self.gpu_buf_of(g, &arc, span);
                let bbuf = self.gpu_buf_of(g, &brc, span);
                match crate::gpu::matmul(g, &abuf, &bbuf, m, k, n) {
                    Ok((cbuf, data)) => {
                        let mut out = TensorData::new(vec![m, n], data);
                        out.device = device;
                        out.gpu_buf = Some(std::rc::Rc::new(cbuf));
                        self.channel.info(
                            span,
                            format!(
                                "matmul [{},{}]·[{},{}] → [{},{}] · output {} B · {} FLOP · gpu (wgpu: {}) · D2H result {} B",
                                m, k, k2, n, m, n, out.bytes(), 2 * m * n * k, g.adapter_name, out.bytes()
                            ),
                        );
                        self.record_tape(span, &mut out, Some(GradNode { op: GradOp::MatMul, inputs: vec![arc, brc] }));
                        return Ok(Value::Tensor(rc_tensor(out)));
                    }
                    Err(e) => self.channel.warn(span, format!("gpu matmul failed ({}) — cpu fallback", e)),
                }
            } else {
                self.channel.info(span, "no gpu adapter — cpu fallback (residency model only)");
            }
        }
        // Real CPU multicore parallelism (large matrices only).
        let (data, threads) = matmul_compute(&a_data, &b_data, m, k, n);
        let mut out = TensorData::new(vec![m, n], data);
        out.device = device;
        let backend = if threads > 1 { format!("cpu×{} threads", threads) } else { "cpu".to_string() };
        self.channel.info(
            span,
            format!("matmul [{},{}]·[{},{}] → [{},{}] · output {} B · {} FLOP · {} · {}", m, k, k2, n, m, n, out.bytes(), 2 * m * n * k, device, backend),
        );
        self.record_tape(span, &mut out, Some(GradNode { op: GradOp::MatMul, inputs: vec![arc, brc] }));
        Ok(Value::Tensor(rc_tensor(out)))
    }

    /// 2D convolution — valid padding, stride 1, differentiable. Honesty: like PyTorch's conv2d this is
    /// *cross-correlation* (the kernel is not flipped); for learned kernels the distinction is immaterial.
    fn conv2d(&mut self, x: &Value, k: &Value, span: Span) -> Result<Value, String> {
        let (xrc, krc) = match (x, k) {
            (Value::Tensor(x), Value::Tensor(k)) => (x.clone(), k.clone()),
            _ => return Err(format!("line {}: conv2d takes two tensors (input, kernel)", span.line)),
        };
        let xt = xrc.borrow();
        let kt = krc.borrow();
        if xt.shape.len() != 2 || kt.shape.len() != 2 {
            return Err(format!("line {}: conv2d takes 2D tensors only (got {}D input, {}D kernel)", span.line, xt.shape.len(), kt.shape.len()));
        }
        let (h, w) = (xt.shape[0], xt.shape[1]);
        let (kh, kw) = (kt.shape[0], kt.shape[1]);
        if kh > h || kw > w {
            self.channel.warn(span, format!("conv2d kernel [{},{}] larger than input [{},{}]", kh, kw, h, w));
            return Err(format!("line {}: conv2d kernel [{},{}] larger than input [{},{}]", span.line, kh, kw, h, w));
        }
        let device = self.combine_device(&xt, &kt, span);
        let (oh, ow) = (h - kh + 1, w - kw + 1);
        let mut data = vec![0f32; oh * ow];
        for i in 0..oh {
            for j in 0..ow {
                let mut s = 0f32;
                for p in 0..kh {
                    for q in 0..kw {
                        s += xt.data[(i + p) * w + (j + q)] * kt.data[p * kw + q];
                    }
                }
                data[i * ow + j] = s;
            }
        }
        drop(xt);
        drop(kt);
        let mut out = TensorData::new(vec![oh, ow], data);
        out.device = device;
        self.channel.info(
            span,
            format!("conv2d [{},{}]⋆[{},{}] → [{},{}] (valid, stride 1) · output {} B · {} FLOP · {}", h, w, kh, kw, oh, ow, out.bytes(), 2 * oh * ow * kh * kw, device),
        );
        self.record_tape(span, &mut out, Some(GradNode { op: GradOp::Conv2d, inputs: vec![xrc, krc] }));
        Ok(Value::Tensor(rc_tensor(out)))
    }

    /// Non-overlapping k×k max pooling (stride = k), differentiable. Trailing rows/cols that don't fill
    /// a window are dropped (standard floor behavior) — the drop is illuminated, not hidden.
    fn maxpool2d(&mut self, x: &Value, k: &Value, span: Span) -> Result<Value, String> {
        let xrc = match x {
            Value::Tensor(x) => x.clone(),
            other => return Err(format!("line {}: maxpool2d(tensor, k) — first arg is {}", span.line, other.type_name())),
        };
        let k = match k {
            Value::Int(n) if *n >= 1 => *n as usize,
            other => return Err(format!("line {}: maxpool2d(tensor, k) — k must be an int ≥ 1 (found {})", span.line, other)),
        };
        let xt = xrc.borrow();
        if xt.shape.len() != 2 {
            return Err(format!("line {}: maxpool2d takes a 2D tensor (got {}D)", span.line, xt.shape.len()));
        }
        let (h, w) = (xt.shape[0], xt.shape[1]);
        if k > h || k > w {
            self.channel.warn(span, format!("maxpool2d window {}×{} larger than input [{},{}]", k, k, h, w));
            return Err(format!("line {}: maxpool2d window {}×{} larger than input [{},{}]", span.line, k, k, h, w));
        }
        let (oh, ow) = (h / k, w / k);
        let mut data = vec![0f32; oh * ow];
        for i in 0..oh {
            for j in 0..ow {
                let mut best = f32::NEG_INFINITY;
                for p in 0..k {
                    for q in 0..k {
                        best = best.max(xt.data[(i * k + p) * w + (j * k + q)]);
                    }
                }
                data[i * ow + j] = best;
            }
        }
        let device = xt.device;
        drop(xt);
        let mut out = TensorData::new(vec![oh, ow], data);
        out.device = device;
        let (dr, dc) = (h - oh * k, w - ow * k);
        let dropped = if dr > 0 || dc > 0 { format!(" · dropped {} row(s), {} col(s)", dr, dc) } else { String::new() };
        self.channel.info(
            span,
            format!("maxpool2d [{},{}] window {}×{} → [{},{}] · output {} B · {}{}", h, w, k, k, oh, ow, out.bytes(), device, dropped),
        );
        self.record_tape(span, &mut out, Some(GradNode { op: GradOp::MaxPool2d { k }, inputs: vec![xrc] }));
        Ok(Value::Tensor(rc_tensor(out)))
    }

    /// The tensor's cached GPU buffer, uploading lazily on first use (v0.50). A resident operand costs
    /// no transfer — that's the §4.3 chain promise, now real.
    #[cfg(feature = "gpu")]
    fn gpu_buf_of(&mut self, g: &'static crate::gpu::Gpu, t: &TensorRef, span: Span) -> std::rc::Rc<wgpu::Buffer> {
        let cached = t.borrow().gpu_buf.clone();
        if let Some(b) = cached {
            self.channel.info(span, "operand resident on device — no transfer");
            return b;
        }
        let data = t.borrow().data.clone();
        let buf = std::rc::Rc::new(crate::gpu::upload(g, &data));
        self.channel.info(span, format!("H2D (lazy): {} B → device", data.len() * 4));
        t.borrow_mut().gpu_buf = Some(buf.clone());
        buf
    }

    /// GPU elementwise tensor∘tensor (v0.51): Some(result) when it ran on the device; None → CPU path.
    /// Same-shape only (broadcast falls back — honest gating).
    #[allow(unused_variables)]
    fn gpu_elementwise_tt(&mut self, arc: &TensorRef, brc: &TensorRef, opcode: u32, rsh: &[usize], device: Device, same_shape: bool, span: Span) -> Option<TensorData> {
        #[cfg(feature = "gpu")]
        if device == Device::Gpu && same_shape {
            if let Some(g) = crate::gpu::ctx() {
                let abuf = self.gpu_buf_of(g, arc, span);
                let bbuf = self.gpu_buf_of(g, brc, span);
                let len: usize = rsh.iter().product();
                match crate::gpu::elementwise(g, &abuf, &bbuf, opcode, 0, 0.0, len) {
                    Ok((cbuf, data)) => {
                        let mut td = TensorData::new(rsh.to_vec(), data);
                        td.device = device;
                        td.gpu_buf = Some(std::rc::Rc::new(cbuf));
                        self.channel.info(span, format!("elementwise · gpu (wgpu: {}) · D2H result {} B", g.adapter_name, len * 4));
                        return Some(td);
                    }
                    Err(e) => self.channel.warn(span, format!("gpu elementwise failed ({}) — cpu fallback", e)),
                }
            }
        }
        None
    }

    /// GPU elementwise tensor∘scalar (v0.51); `scalar_left` = scalar OP tensor (operand order matters for - /).
    #[allow(unused_variables)]
    fn gpu_elementwise_ts(&mut self, arc: &TensorRef, opcode: u32, scalar: f32, scalar_left: bool, span: Span) -> Option<TensorData> {
        #[cfg(feature = "gpu")]
        {
            let (device, shape) = {
                let a = arc.borrow();
                (a.device, a.shape.clone())
            };
            if device == Device::Gpu {
                if let Some(g) = crate::gpu::ctx() {
                    let abuf = self.gpu_buf_of(g, arc, span);
                    let len: usize = shape.iter().product();
                    let mode = if scalar_left { 2 } else { 1 };
                    match crate::gpu::elementwise(g, &abuf, &abuf, opcode, mode, scalar, len) {
                        Ok((cbuf, data)) => {
                            let mut td = TensorData::new(shape, data);
                            td.device = device;
                            td.gpu_buf = Some(std::rc::Rc::new(cbuf));
                            self.channel.info(span, format!("elementwise (scalar) · gpu (wgpu: {}) · D2H result {} B", g.adapter_name, len * 4));
                            return Some(td);
                        }
                        Err(e) => self.channel.warn(span, format!("gpu elementwise failed ({}) — cpu fallback", e)),
                    }
                }
            }
        }
        None
    }

    /// Backprop — starting from a scalar output, walk the tape in reverse topological order accumulating VJPs (A1·A4).
    fn backward(&mut self, output: &TensorRef, span: Span) -> Result<(), String> {
        {
            let o = output.borrow();
            if o.data.len() != 1 {
                return Err(format!("line {}: .backward() works on scalar tensors only (got {} elems)", span.line, o.data.len()));
            }
        }
        // 1. Topological sort (post-order DFS)
        let mut topo: Vec<TensorRef> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = Default::default();
        build_topo(output, &mut visited, &mut topo);
        // 2. Seed: dL/dL = 1
        output.borrow_mut().grad = Some(vec![1.0]);
        // 3. Process in reverse
        let mut grad_bytes = 0usize;
        for node in topo.iter().rev() {
            let (g, gf) = {
                let b = node.borrow();
                (b.grad.clone(), b.grad_fn.clone())
            };
            let g = match g {
                Some(g) => g,
                None => continue,
            };
            grad_bytes += g.len() * 4;
            if let Some(gf) = gf {
                apply_vjp(&gf, &g);
            }
        }
        self.channel.info(span, format!("backprop · {} nodes · grad {} B", topo.len(), grad_bytes));
        Ok(())
    }

    fn tensor_method(&mut self, t: &TensorRef, name: &str, argv: Vec<Value>, span: Span) -> Result<Value, String> {
        let line = span.line;
        // Most tensor methods take no args; sum/mean accept an optional axis, reshape takes one dims array.
        if !matches!(name, "sum" | "mean" | "reshape") {
            check_n(&argv, 0, name, line)?;
        }
        // .gpu()/.cpu() — move residency + illuminate transfer (§4.3, mirrors PyTorch .cuda()/.cpu()).
        if name == "gpu" || name == "cpu" {
            let target = if name == "gpu" { Device::Gpu } else { Device::Host };
            let tb = t.borrow();
            if tb.device == target {
                return Ok(Value::Tensor(rc_tensor(tb.clone()))); // already on that residency — 0 transfer
            }
            let dir = if target == Device::Gpu { "H2D" } else { "D2H" };
            let dst = if target == Device::Gpu { "device" } else { "host" };
            let mut nt = tb.clone();
            nt.device = target;
            // Real backend (`gpu` feature, v0.50): .gpu() actually uploads; .cpu() drops the buffer.
            #[cfg(feature = "gpu")]
            {
                if target == Device::Gpu {
                    if let Some(g) = crate::gpu::ctx() {
                        nt.gpu_buf = Some(std::rc::Rc::new(crate::gpu::upload(g, &nt.data)));
                        self.channel.info(span, format!("{}: {} B → {} (real transfer, wgpu: {})", dir, tb.bytes(), dst, g.adapter_name));
                        return Ok(Value::Tensor(rc_tensor(nt)));
                    }
                } else {
                    nt.gpu_buf = None;
                }
            }
            self.channel.info(span, format!("{}: {} B → {}", dir, tb.bytes(), dst));
            return Ok(Value::Tensor(rc_tensor(nt)));
        }
        match name {
            "backward" => {
                self.backward(t, span)?;
                Ok(Value::Unit)
            }
            "item" => {
                let b = t.borrow();
                if b.data.len() != 1 {
                    return Err(format!("line {}: .item() works on scalar tensors only ({} elems)", line, b.data.len()));
                }
                Ok(Value::Float(b.data[0] as f64))
            }
            "sum" if argv.is_empty() => self.reduce(t, false, span),
            "mean" if argv.is_empty() => self.reduce(t, true, span),
            "sum" => self.reduce_axis(t, false, &argv, span),
            "mean" => self.reduce_axis(t, true, &argv, span),
            "reshape" => {
                let dims: Vec<usize> = match argv.first() {
                    Some(Value::Array(a)) => {
                        let mut v = Vec::new();
                        for el in a.borrow().iter() {
                            match el {
                                Value::Int(n) if *n >= 0 => v.push(*n as usize),
                                other => return Err(format!("line {}: reshape dims must be non-negative ints (found {})", line, other.type_name())),
                            }
                        }
                        v
                    }
                    _ => return Err(format!("line {}: reshape([dims]) takes an array of dims", line)),
                };
                let (data, dev, track) = {
                    let b = t.borrow();
                    (b.data.clone(), b.device, b.requires_grad || b.grad_fn.is_some())
                };
                let total: usize = dims.iter().product();
                if total != data.len() {
                    self.channel.warn(span, format!("reshape {:?} has {} elements but the tensor has {}", dims, total, data.len()));
                    return Err(format!("line {}: reshape to {:?} ({} elements) from {} elements", line, dims, total, data.len()));
                }
                let mut out = TensorData::new(dims.clone(), data);
                out.device = dev;
                self.channel.info(span, format!("reshape → {:?} (view, no copy of layout)", dims));
                if track {
                    self.record_tape(span, &mut out, Some(GradNode { op: GradOp::Reshape, inputs: vec![t.clone()] }));
                }
                Ok(Value::Tensor(rc_tensor(out)))
            }
            "max" => {
                let b = t.borrow();
                if b.data.is_empty() {
                    return Err(format!("line {}: max on empty tensor", line));
                }
                let mx = b.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                Ok(Value::Tensor(rc_tensor(TensorData::new(vec![1], vec![mx])))) // grad not tracked
            }
            _ => Err(format!("line {}: tensor has no method '{}'", line, name)),
        }
    }

    /// Reduction (sum/mean) → scalar tensor (differentiable). Records grad_fn if the input is grad-tracked.
    fn reduce(&mut self, t: &TensorRef, is_mean: bool, span: Span) -> Result<Value, String> {
        let (val, track) = {
            let b = t.borrow();
            if b.data.is_empty() {
                return Err(format!("line {}: reduction on empty tensor", span.line));
            }
            let s: f32 = b.data.iter().sum();
            let v = if is_mean { s / b.data.len() as f32 } else { s };
            (v, b.requires_grad || b.grad_fn.is_some())
        };
        let mut out = TensorData::new(vec![1], vec![val]);
        if track {
            let op = if is_mean { GradOp::Mean } else { GradOp::Sum };
            out.requires_grad = true;
            out.grad_fn = Some(Rc::new(GradNode { op, inputs: vec![t.clone()] }));
            self.channel.info(span, "autodiff graph recorded · reduction");
        }
        Ok(Value::Tensor(rc_tensor(out)))
    }

    /// Axis reduction (`t.sum(axis)` / `t.mean(axis)`) on a 2D tensor → a 1D tensor (differentiable).
    /// axis 0 reduces rows → shape (cols,); axis 1 reduces cols → shape (rows,).
    fn reduce_axis(&mut self, t: &TensorRef, is_mean: bool, argv: &[Value], span: Span) -> Result<Value, String> {
        let axis = match argv.first() {
            Some(Value::Int(a)) if *a == 0 || *a == 1 => *a as usize,
            _ => return Err(format!("line {}: {}(axis) — axis must be 0 or 1", span.line, if is_mean { "mean" } else { "sum" })),
        };
        let (rows, cols, out, track) = {
            let b = t.borrow();
            if b.shape.len() != 2 {
                return Err(format!("line {}: axis reduction needs a 2D tensor (got {}D)", span.line, b.shape.len()));
            }
            let (rows, cols) = (b.shape[0], b.shape[1]);
            let count = if axis == 0 { rows } else { cols } as f32;
            let out_len = if axis == 0 { cols } else { rows };
            let mut out = vec![0.0f32; out_len];
            for r in 0..rows {
                for c in 0..cols {
                    let v = b.data[r * cols + c];
                    out[if axis == 0 { c } else { r }] += v;
                }
            }
            if is_mean {
                for o in &mut out {
                    *o /= count;
                }
            }
            (rows, cols, out, b.requires_grad || b.grad_fn.is_some())
        };
        let out_shape = if axis == 0 { vec![cols] } else { vec![rows] };
        let mut td = TensorData::new(out_shape, out);
        if track {
            let op = if is_mean {
                GradOp::MeanAxis { axis, rows, cols }
            } else {
                GradOp::SumAxis { axis, rows, cols }
            };
            td.requires_grad = true;
            td.grad_fn = Some(Rc::new(GradNode { op, inputs: vec![t.clone()] }));
            self.channel.info(span, format!("autodiff graph recorded · reduction along axis {}", axis));
        }
        Ok(Value::Tensor(rc_tensor(td)))
    }
}

impl Interp {
    fn field(&mut self, recv: Value, name: &str, span: Span) -> Result<Value, String> {
        #[cfg(feature = "ai")]
        if let Value::Tensor(t) = &recv {
            let t = t.borrow();
            return match name {
                "shape" => Ok(Value::array(t.shape.iter().map(|d| Value::Int(*d as i64)).collect())),
                "size" => Ok(Value::Int(t.size() as i64)),
                "ndim" => Ok(Value::Int(t.shape.len() as i64)),
                "dtype" => Ok(Value::Str(t.dtype.to_string())),
                "device" => Ok(Value::Str(t.device.to_string())),
                "grad" => match &t.grad {
                    Some(g) => Ok(Value::Tensor(rc_tensor(TensorData::new(t.shape.clone(), g.clone())))),
                    None => Err(format!("line {}: no grad (run .backward() on the param first)", span.line)),
                },
                _ => Err(format!("line {}: tensor has no attribute '{}'", span.line, name)),
            };
        }
        if let Value::Struct { fields, name: sname } = &recv {
            return fields
                .borrow()
                .get(name)
                .cloned()
                .ok_or_else(|| format!("line {}: {} has no field '{}'", span.line, sname, name));
        }
        match (name, &recv) {
            ("len", _) => length(&recv)
                .map(Value::Int)
                .ok_or_else(|| format!("line {}: {} has no attribute 'len'", span.line, recv.type_name())),
            _ => Err(format!("line {}: {} has no attribute '{}'", span.line, recv.type_name(), name)),
        }
    }

    // ---- arity helpers ----

    fn arity(&mut self, args: &[Expr], n: usize, name: &str, line: usize) -> Result<(), String> {
        if args.len() != n {
            Err(format!("line {}: {}() expects {} arg(s), got {}", line, name, n, args.len()))
        } else {
            Ok(())
        }
    }

    fn arity_v(&self, argv: &[Value], n: usize, name: &str, line: usize) -> Result<(), String> {
        if argv.len() != n {
            Err(format!("line {}: {}() expects {} arg(s), got {}", line, name, n, argv.len()))
        } else {
            Ok(())
        }
    }

    fn one(&mut self, args: &[Expr], name: &str, line: usize) -> Result<Value, String> {
        self.arity(args, 1, name, line)?;
        self.eval(&args[0])
    }

    fn one_num(&mut self, args: &[Expr], name: &str, line: usize) -> Result<f64, String> {
        let v = self.one(args, name, line)?;
        to_f(&v).ok_or_else(|| format!("line {}: {}() takes numbers only (found {})", line, name, v.type_name()))
    }

    fn one_int(&mut self, args: &[Expr], name: &str, line: usize) -> Result<i64, String> {
        match self.one(args, name, line)? {
            Value::Int(n) => Ok(n),
            other => Err(format!("line {}: {}() takes int only (found {})", line, name, other.type_name())),
        }
    }
}

// ---- free functions ----

/// Auto-type a `cin` input token: integer → int, else decimal → float, else string. (C++ cin convenience, dynamic.)
/// (pub: shared with the bytecode VM.)
pub fn parse_input_token(t: &str) -> Value {
    if let Ok(i) = t.parse::<i64>() {
        Value::Int(i)
    } else if let Ok(f) = t.parse::<f64>() {
        Value::Float(f)
    } else {
        Value::Str(t.to_string())
    }
}

fn to_f(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

#[cfg(feature = "ai")]
fn dims_str(shape: &[usize]) -> String {
    shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")
}

/// For softmax/row-wise ops: (rows, cols). 1D=(1,n), 2D=(r,c), otherwise=(1,total).
#[cfg(feature = "ai")]
fn rows_cols(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0]),
        2 => (shape[0], shape[1]),
        _ => (1, shape.iter().product()),
    }
}

/// Row-wise softmax (numerically stable: subtract the row max).
#[cfg(feature = "ai")]
fn softmax_rows(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; data.len()];
    for r in 0..rows {
        let off = r * cols;
        let m = data[off..off + cols].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for i in 0..cols {
            let e = (data[off + i] - m).exp();
            out[off + i] = e;
            sum += e;
        }
        for i in 0..cols {
            out[off + i] /= sum;
        }
    }
    out
}

/// Matrix multiply. Large matrices run on CPU multicore in parallel (output rows split across threads). Returns: (data, thread count).
/// The CPU realization of §4.2 parallel execution. (The real GPU backend would go here — for now, CPU.)
#[cfg(feature = "ai")]
fn matmul_compute(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> (Vec<f32>, usize) {
    let mut out = vec![0f32; m * n];
    let cores = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    // Small matrices: thread overhead dominates, so use a single thread.
    let threads = if m * n * k >= 100_000 && cores > 1 { cores.min(m) } else { 1 };
    if threads <= 1 {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0f32;
                for l in 0..k {
                    s += a[i * k + l] * b[l * n + j];
                }
                out[i * n + j] = s;
            }
        }
        return (out, 1);
    }
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|scope| {
        for (t, chunk) in out.chunks_mut(rows_per * n).enumerate() {
            let row0 = t * rows_per;
            scope.spawn(move || {
                let rows = chunk.len() / n;
                for ri in 0..rows {
                    let i = row0 + ri;
                    for j in 0..n {
                        let mut s = 0f32;
                        for l in 0..k {
                            s += a[i * k + l] * b[l * n + j];
                        }
                        chunk[ri * n + j] = s;
                    }
                }
            });
        }
    });
    (out, threads)
}

// ---- broadcasting (NumPy rules) ----

/// Broadcast result shape of two shapes (right-aligned, each axis equal or one side is 1).
#[cfg(feature = "ai")]
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let d = a.len().max(b.len());
    let mut out = vec![0usize; d];
    for i in 0..d {
        let ai = if i < d - a.len() { 1 } else { a[i - (d - a.len())] };
        let bi = if i < d - b.len() { 1 } else { b[i - (d - b.len())] };
        out[i] = if ai == bi || bi == 1 {
            ai
        } else if ai == 1 {
            bi
        } else {
            return None;
        };
    }
    Some(out)
}

#[cfg(feature = "ai")]
fn unravel(mut r: usize, shape: &[usize]) -> Vec<usize> {
    let mut mi = vec![0usize; shape.len()];
    for ax in (0..shape.len()).rev() {
        mi[ax] = r % shape[ax];
        r /= shape[ax];
    }
    mi
}

/// Maps the result multi-index mi (length d) to the operand's flat index (shape, right-aligned) — axes of size 1 collapse to 0.
#[cfg(feature = "ai")]
fn idx_in(mi: &[usize], shape: &[usize], d: usize) -> usize {
    let off = d - shape.len();
    let mut idx = 0;
    let mut stride = 1;
    for ax in (0..shape.len()).rev() {
        let coord = if shape[ax] == 1 { 0 } else { mi[off + ax] };
        idx += coord * stride;
        stride *= shape[ax];
    }
    idx
}

#[cfg(feature = "ai")]
fn bcast_binary(a: &[f32], ash: &[usize], b: &[f32], bsh: &[usize], rsh: &[usize], f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let d = rsh.len();
    let total: usize = rsh.iter().product();
    (0..total)
        .map(|r| {
            let mi = unravel(r, rsh);
            f(a[idx_in(&mi, ash, d)], b[idx_in(&mi, bsh, d)])
        })
        .collect()
}

/// Reduces the result grad g (shape gsh) down to the target shape (summing over broadcast axes).
#[cfg(feature = "ai")]
fn reduce_to(g: &[f32], gsh: &[usize], target: &[usize]) -> Vec<f32> {
    let d = gsh.len();
    let tsize: usize = target.iter().product::<usize>().max(1);
    let mut out = vec![0f32; tsize];
    for (r, &gv) in g.iter().enumerate() {
        let mi = unravel(r, gsh);
        out[idx_in(&mi, target, d)] += gv;
    }
    out
}

// ---- automatic differentiation backprop (free fns) ----

/// Topological sort via post-order DFS — inputs first, node last. (Visited marked by Rc pointer.)
#[cfg(feature = "ai")]
fn build_topo(t: &TensorRef, visited: &mut std::collections::HashSet<usize>, topo: &mut Vec<TensorRef>) {
    let id = Rc::as_ptr(t) as usize;
    if !visited.insert(id) {
        return;
    }
    let inputs = t.borrow().grad_fn.as_ref().map(|gf| gf.inputs.clone()).unwrap_or_default();
    for inp in &inputs {
        build_topo(inp, visited, topo);
    }
    topo.push(t.clone());
}

/// Distributes a node's grad `g` to its inputs (VJP) — accumulating into each input's .grad.
#[cfg(feature = "ai")]
fn apply_vjp(gf: &GradNode, g: &[f32]) {
    match &gf.op {
        GradOp::Add => {
            let ash = gf.inputs[0].borrow().shape.clone();
            let bsh = gf.inputs[1].borrow().shape.clone();
            let gsh = broadcast_shapes(&ash, &bsh).unwrap_or_else(|| ash.clone());
            accumulate(&gf.inputs[0], &reduce_to(g, &gsh, &ash));
            accumulate(&gf.inputs[1], &reduce_to(g, &gsh, &bsh));
        }
        GradOp::Sub => {
            let ash = gf.inputs[0].borrow().shape.clone();
            let bsh = gf.inputs[1].borrow().shape.clone();
            let gsh = broadcast_shapes(&ash, &bsh).unwrap_or_else(|| ash.clone());
            accumulate(&gf.inputs[0], &reduce_to(g, &gsh, &ash));
            let neg: Vec<f32> = g.iter().map(|x| -x).collect();
            accumulate(&gf.inputs[1], &reduce_to(&neg, &gsh, &bsh));
        }
        GradOp::MulElem => {
            let (a_data, ash) = {
                let a = gf.inputs[0].borrow();
                (a.data.clone(), a.shape.clone())
            };
            let (b_data, bsh) = {
                let b = gf.inputs[1].borrow();
                (b.data.clone(), b.shape.clone())
            };
            let gsh = broadcast_shapes(&ash, &bsh).unwrap_or_else(|| ash.clone());
            let d = gsh.len();
            let ga_full: Vec<f32> = (0..g.len()).map(|r| { let mi = unravel(r, &gsh); g[r] * b_data[idx_in(&mi, &bsh, d)] }).collect();
            let gb_full: Vec<f32> = (0..g.len()).map(|r| { let mi = unravel(r, &gsh); g[r] * a_data[idx_in(&mi, &ash, d)] }).collect();
            accumulate(&gf.inputs[0], &reduce_to(&ga_full, &gsh, &ash));
            accumulate(&gf.inputs[1], &reduce_to(&gb_full, &gsh, &bsh));
        }
        GradOp::ScalarMul(s) => {
            let gs: Vec<f32> = g.iter().map(|x| x * s).collect();
            accumulate(&gf.inputs[0], &gs);
        }
        GradOp::MatMul => {
            let (a_data, a_shape) = {
                let a = gf.inputs[0].borrow();
                (a.data.clone(), a.shape.clone())
            };
            let (b_data, b_shape) = {
                let b = gf.inputs[1].borrow();
                (b.data.clone(), b.shape.clone())
            };
            let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
            let ga = matmul_bt(g, &b_data, m, n, k); // g(m,n) @ Bᵀ → (m,k)
            let gb = matmul_at(&a_data, g, m, k, n); // Aᵀ @ g(m,n) → (k,n)
            accumulate(&gf.inputs[0], &ga);
            accumulate(&gf.inputs[1], &gb);
        }
        GradOp::Sum => {
            let size = gf.inputs[0].borrow().data.len();
            accumulate(&gf.inputs[0], &vec![g[0]; size]);
        }
        GradOp::Mean => {
            let size = gf.inputs[0].borrow().data.len();
            accumulate(&gf.inputs[0], &vec![g[0] / size as f32; size]);
        }
        GradOp::Reshape => {
            // Reshape keeps the row-major data order, so the gradient flows back unchanged (the input's
            // flat layout equals the output's). accumulate adds it onto the input's grad in input shape.
            accumulate(&gf.inputs[0], g);
        }
        GradOp::Conv2d => {
            // Forward: out[i,j] = Σ x[i+p, j+q]·k[p,q] (valid cross-correlation).
            // dX[a,b] = Σ_{p,q} k[p,q]·g[a-p, b-q] (where the g index is in range) — a "full" correlation.
            // dK[p,q] = Σ_{i,j} g[i,j]·x[i+p, j+q] — a valid correlation of x with g.
            let (x_data, x_shape) = {
                let x = gf.inputs[0].borrow();
                (x.data.clone(), x.shape.clone())
            };
            let (k_data, k_shape) = {
                let k = gf.inputs[1].borrow();
                (k.data.clone(), k.shape.clone())
            };
            let (h, w) = (x_shape[0], x_shape[1]);
            let (kh, kw) = (k_shape[0], k_shape[1]);
            let (oh, ow) = (h - kh + 1, w - kw + 1);
            let mut gx = vec![0f32; h * w];
            for a in 0..h {
                for b in 0..w {
                    let mut s = 0f32;
                    for p in 0..kh {
                        for q in 0..kw {
                            if a >= p && b >= q && a - p < oh && b - q < ow {
                                s += k_data[p * kw + q] * g[(a - p) * ow + (b - q)];
                            }
                        }
                    }
                    gx[a * w + b] = s;
                }
            }
            let mut gk = vec![0f32; kh * kw];
            for p in 0..kh {
                for q in 0..kw {
                    let mut s = 0f32;
                    for i in 0..oh {
                        for j in 0..ow {
                            s += g[i * ow + j] * x_data[(i + p) * w + (j + q)];
                        }
                    }
                    gk[p * kw + q] = s;
                }
            }
            accumulate(&gf.inputs[0], &gx);
            accumulate(&gf.inputs[1], &gk);
        }
        GradOp::MaxPool2d { k } => {
            // The max element of each window gets that window's gradient; everything else gets 0.
            // The argmax is recomputed from the saved input (first hit wins on ties, matching forward).
            let k = *k;
            let (x_data, x_shape) = {
                let x = gf.inputs[0].borrow();
                (x.data.clone(), x.shape.clone())
            };
            let (h, w) = (x_shape[0], x_shape[1]);
            let (oh, ow) = (h / k, w / k);
            let mut gx = vec![0f32; h * w];
            for i in 0..oh {
                for j in 0..ow {
                    let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
                    for p in 0..k {
                        for q in 0..k {
                            let idx = (i * k + p) * w + (j * k + q);
                            if x_data[idx] > best {
                                best = x_data[idx];
                                bi = idx;
                            }
                        }
                    }
                    gx[bi] += g[i * ow + j];
                }
            }
            accumulate(&gf.inputs[0], &gx);
        }
        GradOp::SumAxis { axis, rows, cols } | GradOp::MeanAxis { axis, rows, cols } => {
            // Broadcast the reduced-axis gradient back to the input shape. Mean also divides by count.
            let (axis, rows, cols) = (*axis, *rows, *cols);
            let is_mean = matches!(gf.op, GradOp::MeanAxis { .. });
            let count = if axis == 0 { rows } else { cols } as f32;
            let mut gi = vec![0.0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    let gv = g[if axis == 0 { c } else { r }];
                    gi[r * cols + c] = if is_mean { gv / count } else { gv };
                }
            }
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Relu => {
            let mask: Vec<f32> = gf.inputs[0].borrow().data.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
            let gi: Vec<f32> = g.iter().zip(&mask).map(|(gx, m)| gx * m).collect();
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Sigmoid => {
            let x = gf.inputs[0].borrow().data.clone();
            let gi: Vec<f32> = g.iter().zip(&x).map(|(gx, xi)| { let s = 1.0 / (1.0 + (-xi).exp()); gx * s * (1.0 - s) }).collect();
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Tanh => {
            let x = gf.inputs[0].borrow().data.clone();
            let gi: Vec<f32> = g.iter().zip(&x).map(|(gx, xi)| { let t = xi.tanh(); gx * (1.0 - t * t) }).collect();
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Exp => {
            let x = gf.inputs[0].borrow().data.clone();
            let gi: Vec<f32> = g.iter().zip(&x).map(|(gx, xi)| gx * xi.exp()).collect();
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Log => {
            let x = gf.inputs[0].borrow().data.clone();
            let gi: Vec<f32> = g.iter().zip(&x).map(|(gx, xi)| gx / xi).collect();
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Softmax => {
            let (x, shape) = {
                let inp = gf.inputs[0].borrow();
                (inp.data.clone(), inp.shape.clone())
            };
            let (rows, cols) = rows_cols(&shape);
            let s = softmax_rows(&x, rows, cols);
            let mut gi = vec![0f32; x.len()];
            for r in 0..rows {
                let off = r * cols;
                let dot: f32 = (0..cols).map(|j| g[off + j] * s[off + j]).sum();
                for i in 0..cols {
                    gi[off + i] = s[off + i] * (g[off + i] - dot);
                }
            }
            accumulate(&gf.inputs[0], &gi);
        }
        GradOp::Transpose => {
            let (m, n) = {
                let inp = gf.inputs[0].borrow();
                (inp.shape[0], inp.shape[1])
            };
            let mut gi = vec![0f32; m * n]; // g is (n,m); gi[i,j] = g[j,i]
            for i in 0..m {
                for j in 0..n {
                    gi[i * n + j] = g[j * m + i];
                }
            }
            accumulate(&gf.inputs[0], &gi);
        }
    }
}

#[cfg(feature = "ai")]
fn accumulate(t: &TensorRef, contrib: &[f32]) {
    let mut tb = t.borrow_mut();
    match &mut tb.grad {
        Some(g) => {
            for (x, c) in g.iter_mut().zip(contrib) {
                *x += c;
            }
        }
        None => tb.grad = Some(contrib.to_vec()),
    }
}

#[cfg(feature = "ai")]
fn accumulate_scaled(t: &TensorRef, contrib: &[f32], s: f32) {
    let v: Vec<f32> = contrib.iter().map(|x| x * s).collect();
    accumulate(t, &v);
}

/// g(m,n) @ Bᵀ(n,k) → (m,k), B is (k,n) row-major.
#[cfg(feature = "ai")]
fn matmul_bt(g: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * k];
    for i in 0..m {
        for l in 0..k {
            let mut s = 0f32;
            for j in 0..n {
                s += g[i * n + j] * b[l * n + j];
            }
            out[i * k + l] = s;
        }
    }
    out
}

/// Aᵀ(k,m) @ g(m,n) → (k,n), A is (m,k) row-major.
#[cfg(feature = "ai")]
fn matmul_at(a: &[f32], g: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; k * n];
    for l in 0..k {
        for j in 0..n {
            let mut s = 0f32;
            for i in 0..m {
                s += a[i * k + l] * g[i * n + j];
            }
            out[l * n + j] = s;
        }
    }
    out
}

#[cfg(feature = "ai")]
fn rc_tensor(td: TensorData) -> TensorRef {
    std::rc::Rc::new(std::cell::RefCell::new(td))
}

/// Nested array/scalar → tensor (infer shape + flatten to f32).
#[cfg(feature = "ai")]
fn build_tensor(v: &Value) -> Result<TensorData, String> {
    let mut shape = Vec::new();
    let mut data = Vec::new();
    infer_tensor(v, &mut shape, &mut data, 0)?;
    let expected: usize = shape.iter().product();
    if data.len() != expected.max(1) && !shape.is_empty() {
        return Err("ragged tensor (shape inference failed)".into());
    }
    Ok(TensorData::new(shape, data))
}

#[cfg(feature = "ai")]
fn infer_tensor(v: &Value, shape: &mut Vec<usize>, data: &mut Vec<f32>, depth: usize) -> Result<(), String> {
    match v {
        Value::Array(a) => {
            let b = a.borrow();
            if depth == shape.len() {
                shape.push(b.len());
            } else if shape[depth] != b.len() {
                return Err("ragged tensor (shape mismatch)".into());
            }
            for el in b.iter() {
                infer_tensor(el, shape, data, depth + 1)?;
            }
            Ok(())
        }
        Value::Int(n) => {
            data.push(*n as f32);
            Ok(())
        }
        Value::Float(x) => {
            data.push(*x as f32);
            Ok(())
        }
        other => Err(format!("tensor elements must be numbers only ({})", other.type_name())),
    }
}

/// Integer array → shape vector (for zeros/ones).
#[cfg(feature = "ai")]
fn shape_from(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(a) => a
            .borrow()
            .iter()
            .map(|d| match d {
                Value::Int(n) if *n >= 0 => Ok(*n as usize),
                _ => Err("shape must be an array of non-negative integers".to_string()),
            })
            .collect(),
        _ => Err("shape must be an integer array (e.g. [2, 3])".into()),
    }
}

/// If the pattern matches the value, returns the list of variable bindings, otherwise None.
/// (pub: shared with the bytecode VM so match semantics stay identical.)
/// Every identifier a closure body might read (v0.42) — an *over-approximation* (all Ident uses + call
/// names, shadowing ignored). Intersected with the visible bindings at lambda creation to decide what
/// gets captured; over-capturing is harmless (the frame value is simply overwritten before use).
/// Pub: the borrow static pass (check, v0.49) also uses it for closure-capture detection.
pub fn collect_free_idents(body: &[Stmt], into: &mut std::collections::HashSet<String>) {
    collect_idents_stmts(body, into)
}

fn collect_idents_stmts(body: &[Stmt], into: &mut std::collections::HashSet<String>) {
    for s in body {
        match s {
            Stmt::Let { value, .. } => collect_idents_expr(value, into),
            Stmt::Assign { target, value, .. } => {
                collect_idents_expr(target, into);
                collect_idents_expr(value, into);
            }
            Stmt::Expr(e) => collect_idents_expr(e, into),
            Stmt::Return(opt, _) => {
                if let Some(e) = opt {
                    collect_idents_expr(e, into);
                }
            }
            Stmt::If { branches, else_body, .. } => {
                for (c, b) in branches {
                    collect_idents_expr(c, into);
                    collect_idents_stmts(b, into);
                }
                if let Some(b) = else_body {
                    collect_idents_stmts(b, into);
                }
            }
            Stmt::While { cond, body, .. } => {
                collect_idents_expr(cond, into);
                collect_idents_stmts(body, into);
            }
            Stmt::For { iter, body, .. } => {
                collect_idents_expr(iter, into);
                collect_idents_stmts(body, into);
            }
            Stmt::Match { subject, arms, .. } => {
                collect_idents_expr(subject, into);
                for arm in arms {
                    collect_idents_stmts(&arm.body, into);
                }
            }
            Stmt::Cout(parts, _) => {
                for e in parts {
                    collect_idents_expr(e, into);
                }
            }
            Stmt::Cin(targets, _) => {
                for e in targets {
                    collect_idents_expr(e, into);
                }
            }
            Stmt::ShowProvenance(e, _) => collect_idents_expr(e, into),
            Stmt::Trust(inner, _) => collect_idents_stmts(std::slice::from_ref(inner), into),
            Stmt::Fn { body, .. } | Stmt::Impl { methods: body, .. } => collect_idents_stmts(body, into),
            Stmt::Break(_) | Stmt::Continue(_) | Stmt::Import(..) | Stmt::Struct { .. } | Stmt::Enum { .. } => {}
        }
    }
}

fn collect_idents_expr(e: &Expr, into: &mut std::collections::HashSet<String>) {
    match e {
        Expr::Ident(n, _) => {
            into.insert(n.clone());
        }
        Expr::Call(name, args, _) => {
            into.insert(name.clone()); // may be a variable holding a fn value
            for a in args {
                collect_idents_expr(a, into);
            }
        }
        Expr::CallValue(callee, args, _) => {
            collect_idents_expr(callee, into);
            for a in args {
                collect_idents_expr(a, into);
            }
        }
        Expr::Method(recv, _, args, _) => {
            collect_idents_expr(recv, into);
            for a in args {
                collect_idents_expr(a, into);
            }
        }
        Expr::Array(elems, _) => {
            for el in elems {
                collect_idents_expr(el, into);
            }
        }
        Expr::Range(a, b, _) | Expr::And(a, b, _) | Expr::Or(a, b, _) | Expr::Binary(_, a, b, _) | Expr::Index(a, b, _) => {
            collect_idents_expr(a, into);
            collect_idents_expr(b, into);
        }
        Expr::Neg(a, _) | Expr::Not(a, _) | Expr::Try(a, _) | Expr::AddrOf(a, _) | Expr::AddrOfMut(a, _)
        | Expr::Deref(a, _) | Expr::Field(a, _, _) => collect_idents_expr(a, into),
        Expr::StructLit(_, fields, _) => {
            for (_, v) in fields {
                collect_idents_expr(v, into);
            }
        }
        Expr::EnumLit(_, _, args, _) => {
            for a in args {
                collect_idents_expr(a, into);
            }
        }
        Expr::Match(subject, arms, _) => {
            collect_idents_expr(subject, into);
            for (_, body) in arms {
                collect_idents_expr(body, into);
            }
        }
        Expr::Lambda(_, body, _) => collect_idents_stmts(body, into), // nested lambda reads count too
        Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::Str(..) | Expr::Map(_) => {}
    }
}

pub fn match_pattern(pat: &Pattern, val: &Value) -> Option<Vec<(String, Value)>> {
    match pat {
        Pattern::Wildcard => Some(vec![]),
        Pattern::Bind(n) => Some(vec![(n.clone(), val.clone())]),
        Pattern::Int(n) => matches!(val, Value::Int(m) if m == n).then(Vec::new),
        Pattern::Bool(b) => matches!(val, Value::Bool(c) if c == b).then(Vec::new),
        Pattern::Str(s) => matches!(val, Value::Str(t) if t == s).then(Vec::new),
        Pattern::Enum(ename, variant, subpats) => {
            if let Value::Enum { name, variant: v, payload } = val {
                if name == ename && v == variant && payload.len() == subpats.len() {
                    let mut binds = Vec::new();
                    for (sp, pv) in subpats.iter().zip(payload) {
                        binds.extend(match_pattern(sp, pv)?);
                    }
                    return Some(binds);
                }
            }
            None
        }
        Pattern::Struct(sname, fieldpats) => {
            if let Value::Struct { name, fields } = val {
                if name != sname {
                    return None;
                }
                // Clone the field values first (move the borrow out of the recursion), then match.
                let vals: Vec<Value> = {
                    let f = fields.borrow();
                    let mut vs = Vec::new();
                    for (fname, _) in fieldpats {
                        vs.push(f.get(fname)?.clone());
                    }
                    vs
                };
                let mut binds = Vec::new();
                for ((_, fp), fv) in fieldpats.iter().zip(vals) {
                    binds.extend(match_pattern(fp, &fv)?);
                }
                return Some(binds);
            }
            None
        }
    }
}

fn length(v: &Value) -> Option<i64> {
    match v {
        Value::Array(a) => Some(a.borrow().len() as i64),
        Value::Str(s) => Some(s.chars().count() as i64),
        Value::Map(m) => Some(m.borrow().len() as i64),
        Value::Heap(h) => Some(h.borrow().len() as i64),
        Value::Set(s) => Some(s.borrow().len() as i64),
        Value::StrBuf(b) => Some(b.borrow().chars().count() as i64),
        Value::Range(a, b) => Some((b - a).max(0)),
        _ => None,
    }
}

fn check_n(argv: &[Value], n: usize, name: &str, line: usize) -> Result<(), String> {
    if argv.len() != n {
        Err(format!("line {}: {}() expects {} arg(s), got {}", line, name, n, argv.len()))
    } else {
        Ok(())
    }
}

fn as_index(v: &Value, len: usize, line: usize) -> Result<usize, String> {
    match v {
        Value::Int(n) if *n >= 0 && (*n as usize) < len => Ok(*n as usize),
        Value::Int(n) => Err(format!("line {}: index {} out of range (len {})", line, n, len)),
        other => Err(format!("line {}: index must be int (found {})", line, other.type_name())),
    }
}

fn sum_array(xs: &[Value], line: usize) -> Result<Value, String> {
    let all_int = xs.iter().all(|v| matches!(v, Value::Int(_)));
    if all_int {
        Ok(Value::Int(xs.iter().map(|v| if let Value::Int(n) = v { *n } else { 0 }).sum()))
    } else {
        let mut s = 0f64;
        for v in xs {
            s += to_f(v).ok_or_else(|| format!("line {}: sum() takes a number array only (element {})", line, v.type_name()))?;
        }
        Ok(Value::Float(s))
    }
}

fn to_int(argv: &[Value], line: usize) -> Result<Value, String> {
    match argv {
        [v] => match v {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(x) => Ok(Value::Int(*x as i64)),
            Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
            Value::Str(s) => s
                .trim()
                .parse()
                .map(Value::Int)
                .map_err(|_| format!("line {}: failed to parse int '{}'", line, s)),
            other => Err(format!("line {}: int() cannot be applied to {}", line, other.type_name())),
        },
        [Value::Str(s), Value::Int(base)] => {
            let b = u32::try_from(*base).map_err(|_| format!("line {}: invalid radix {}", line, base))?;
            let t = s.trim().trim_start_matches("0x").trim_start_matches("0b");
            i64::from_str_radix(t, b)
                .map(Value::Int)
                .map_err(|_| format!("line {}: int('{}', {}) failed to parse", line, s, base))
        }
        _ => Err(format!("line {}: int(x) or int(str, base)", line)),
    }
}

fn pow(b: &Value, e: &Value, line: usize) -> Result<Value, String> {
    if let (Value::Int(base), Value::Int(exp)) = (b, e) {
        if *exp >= 0 {
            if let Some(v) = base.checked_pow(*exp as u32) {
                return Ok(Value::Int(v));
            }
        }
    }
    match (to_f(b), to_f(e)) {
        (Some(x), Some(y)) => Ok(Value::Float(x.powf(y))),
        _ => Err(format!("line {}: pow() takes numbers only", line)),
    }
}

fn radix_str(n: i64, radix: u32, prefix: &str) -> String {
    let digits = |mut x: u64| -> String {
        if x == 0 {
            return "0".into();
        }
        let mut out = Vec::new();
        while x > 0 {
            let d = (x % radix as u64) as u32;
            out.push(std::char::from_digit(d, radix).unwrap());
            x /= radix as u64;
        }
        out.iter().rev().collect()
    };
    if n < 0 {
        format!("-{}{}", prefix, digits((-n) as u64))
    } else {
        format!("{}{}", prefix, digits(n as u64))
    }
}

/// Min-heap push (sift-up).
pub fn heap_push(v: &mut Vec<Value>, x: Value) -> Result<(), String> {
    v.push(x);
    let mut i = v.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if value_cmp(&v[i], &v[parent])? == Ordering::Less {
            v.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
    Ok(())
}

/// Min-heap pop (returns the minimum, sift-down).
pub fn heap_pop(v: &mut Vec<Value>) -> Result<Option<Value>, String> {
    if v.is_empty() {
        return Ok(None);
    }
    let n = v.len();
    v.swap(0, n - 1);
    let min = v.pop();
    let len = v.len();
    let mut i = 0;
    loop {
        let (l, r) = (2 * i + 1, 2 * i + 2);
        let mut smallest = i;
        if l < len && value_cmp(&v[l], &v[smallest])? == Ordering::Less {
            smallest = l;
        }
        if r < len && value_cmp(&v[r], &v[smallest])? == Ordering::Less {
            smallest = r;
        }
        if smallest == i {
            break;
        }
        v.swap(i, smallest);
        i = smallest;
    }
    Ok(min)
}

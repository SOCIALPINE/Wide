//! Cranelift JIT backend (stage 3 of the roadmap, DESIGN §2) — the first *real native code*.
//!
//! Scope (v0.37 + v0.41 + v0.47): compiles numeric functions to machine code — params, `let`/
//! assignment, `+ - *`, comparisons, `and`/`or`/`not`, `if`/`elif`/`else`, `while`, `break`,
//! `continue`, `return` — over **i64 or f64**. Each function gets one ABI: **I64** (all params and the
//! return are ints — tried first) or **F64** (params seeded float; int literals inside are promoted
//! per-expression, like the tree-walker's numeric promotion). Calls between JIT functions (v0.41),
//! including direct and mutual recursion, work within one ABI; cross-ABI calls fall back. Eligibility
//! is a *fixed point* over the whole program's functions. Anything else (strings, collections, tensors,
//! calls to interpreter-only functions, division) stays on the tree-walker (the reference).
//! Division is excluded on purpose — for ints native `sdiv` traps on /0, for floats native fdiv gives
//! `inf` — where the interpreter reports a checked error in *both* cases (parity first).
//! Honesty: deep native recursion can overflow the process stack — there is no guard (yet).
//!
//! This is gated behind the `jit` Cargo feature (the project's first external dependency).

use std::collections::HashMap;

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};

use crate::ast::{BinOp, Expr, Stmt};

/// One user function as the JIT sees it: (name, params, body).
pub type FnDef = (String, Vec<String>, Vec<Stmt>);

/// A JIT function's ABI — every parameter and the return value share one machine type.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Abi {
    I64,
    F64,
}

/// Value kind in the JIT subset — a native result must mean exactly what the tree-walker produces.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Int,
    Float,
    Bool,
    Poison, // conflicting assignments (e.g. int then bool) — reads of it reject the function
}

fn join(a: Kind, b: Kind) -> Kind {
    match (a, b) {
        (x, y) if x == y => x,
        (Kind::Int, Kind::Float) | (Kind::Float, Kind::Int) => Kind::Float, // numeric promotion
        _ => Kind::Poison,
    }
}

/// Classify every function: I64 tried first (ints stay ints), then F64. The fixed point then drops
/// functions whose calls leave the set, mismatch arity, or cross ABIs (analysis assumes callee ABI ==
/// caller ABI). The caller must pre-filter builtin-shadowing names (builtins dispatch first).
pub fn eligible_set(funcs: &[FnDef]) -> HashMap<String, (Abi, bool)> {
    let arities: HashMap<&str, usize> = funcs.iter().map(|(n, p, _)| (n.as_str(), p.len())).collect();
    let mut set: HashMap<String, (Abi, bool)> = HashMap::new();
    for (n, params, body) in funcs {
        if params.len() > 8 {
            continue;
        }
        if let Some((_, ret_bool)) = classify(params, body, Abi::I64) {
            set.insert(n.clone(), (Abi::I64, ret_bool));
        } else if let Some((_, ret_bool)) = classify(params, body, Abi::F64) {
            set.insert(n.clone(), (Abi::F64, ret_bool));
        }
    }
    let calls: HashMap<&str, Vec<(String, usize)>> = funcs
        .iter()
        .map(|(n, _, body)| {
            let mut cs = Vec::new();
            collect_calls(body, &mut cs);
            (n.as_str(), cs)
        })
        .collect();
    loop {
        let drops: Vec<String> = set
            .keys()
            .filter(|name| {
                let my_abi = set[name.as_str()].0;
                calls[name.as_str()].iter().any(|(callee, argc)| {
                    // callees must share the ABI and return a number (analysis assumed a numeric
                    // result); a bool-returning callee is fine to *dispatch* but not to call natively.
                    set.get(callee) != Some(&(my_abi, false)) || arities.get(callee.as_str()) != Some(argc)
                })
            })
            .cloned()
            .collect();
        if drops.is_empty() {
            return set;
        }
        for d in drops {
            set.remove(&d);
        }
    }
}

/// Can this function compile with the given ABI? Returns its variable kinds and whether the result
/// is a wide bool (machine 0/1, re-wrapped at dispatch — v0.55). Return kinds must be uniform:
/// I64 → all Int or all Bool; F64 → all Float/Int (promoted). Calls are assumed to target same-ABI,
/// number-returning functions — the fixed point enforces it.
fn classify(params: &[String], body: &[Stmt], abi: Abi) -> Option<(HashMap<String, Kind>, bool)> {
    if !body.iter().all(stmt_ok) || !always_returns(body) {
        return None;
    }
    let kinds = var_kinds(params, body, abi);
    if !validate(body, &kinds, abi) {
        return None;
    }
    let mut rets = Vec::new();
    collect_return_kinds(body, &kinds, abi, &mut rets);
    let ret_bool = match abi {
        Abi::I64 => {
            if rets.iter().all(|k| *k == Kind::Int) {
                false
            } else if rets.iter().all(|k| *k == Kind::Bool) {
                true
            } else {
                return None; // mixed int/bool returns — the machine word would be ambiguous
            }
        }
        Abi::F64 => {
            if rets.iter().all(|k| matches!(k, Kind::Int | Kind::Float)) {
                false
            } else {
                return None; // bool returns stay on the I64 ABI (one transmute family per ABI)
            }
        }
    };
    Some((kinds, ret_bool))
}

fn collect_return_kinds(body: &[Stmt], vars: &HashMap<String, Kind>, abi: Abi, out: &mut Vec<Kind>) {
    for s in body {
        match s {
            Stmt::Return(Some(e), _) => {
                if let Some(k) = expr_kind(e, vars, abi) {
                    out.push(k);
                }
            }
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    collect_return_kinds(b, vars, abi, out);
                }
                if let Some(b) = else_body {
                    collect_return_kinds(b, vars, abi, out);
                }
            }
            Stmt::While { body, .. } | Stmt::For { body, .. } => collect_return_kinds(body, vars, abi, out),
            _ => {}
        }
    }
}

/// Fixpoint over assignments: each variable's final kind is the join of everything assigned to it
/// (order-insensitive — a var that is ever assigned a float is F64 for its whole life; int stores
/// into it are promoted at codegen). Params are seeded by the ABI.
fn var_kinds(params: &[String], body: &[Stmt], abi: Abi) -> HashMap<String, Kind> {
    let seed = if abi == Abi::I64 { Kind::Int } else { Kind::Float };
    let mut vars: HashMap<String, Kind> = params.iter().map(|p| (p.clone(), seed)).collect();
    loop {
        let before = vars.clone();
        collect_assign_kinds(body, &mut vars, abi);
        if vars == before {
            return vars;
        }
    }
}

fn collect_assign_kinds(body: &[Stmt], vars: &mut HashMap<String, Kind>, abi: Abi) {
    for s in body {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { target: Expr::Ident(name, _), value, .. } => {
                let k = expr_kind(value, vars, abi).unwrap_or(Kind::Poison);
                let nk = match vars.get(name) {
                    Some(prev) => join(*prev, k),
                    None => k,
                };
                vars.insert(name.clone(), nk);
            }
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    collect_assign_kinds(b, vars, abi);
                }
                if let Some(b) = else_body {
                    collect_assign_kinds(b, vars, abi);
                }
            }
            Stmt::While { body, .. } => collect_assign_kinds(body, vars, abi),
            Stmt::For { var, body, .. } => {
                let nk = match vars.get(var) {
                    Some(prev) => join(*prev, Kind::Int),
                    None => Kind::Int,
                };
                vars.insert(var.clone(), nk);
                collect_assign_kinds(body, vars, abi);
            }
            _ => {}
        }
    }
}

/// Full validation with the final kinds: conditions are bool, arithmetic is numeric (no division —
/// /0 parity), calls match the assumed ABI, and every `return` matches the function's ABI.
fn validate(body: &[Stmt], vars: &HashMap<String, Kind>, abi: Abi) -> bool {
    for s in body {
        let ok = match s {
            Stmt::Let { value, .. } | Stmt::Assign { target: Expr::Ident(..), value, .. } => {
                matches!(expr_kind(value, vars, abi), Some(k) if k != Kind::Poison)
            }
            Stmt::Expr(e) => expr_kind(e, vars, abi).is_some(),
            Stmt::Return(Some(e), _) => match (abi, expr_kind(e, vars, abi)) {
                (Abi::I64, Some(Kind::Int)) | (Abi::I64, Some(Kind::Bool)) => true, // bool re-wrapped at dispatch (v0.55)
                (Abi::F64, Some(Kind::Float)) | (Abi::F64, Some(Kind::Int)) => true, // int promoted
                _ => false,
            },
            Stmt::Return(None, _) => false, // bare return → Unit on the tree-walker, a number natively
            Stmt::If { branches, else_body, .. } => {
                branches
                    .iter()
                    .all(|(c, b)| expr_kind(c, vars, abi) == Some(Kind::Bool) && validate(b, vars, abi))
                    && else_body.as_ref().map_or(true, |b| validate(b, vars, abi))
            }
            Stmt::While { cond, body, .. } => {
                expr_kind(cond, vars, abi) == Some(Kind::Bool) && validate(body, vars, abi)
            }
            Stmt::For { iter: Expr::Range(lo, hi, _), body, .. } => {
                expr_kind(lo, vars, abi) == Some(Kind::Int)
                    && expr_kind(hi, vars, abi) == Some(Kind::Int)
                    && validate(body, vars, abi)
            }
            Stmt::Break(_) | Stmt::Continue(_) => true,
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    true
}

fn expr_kind(e: &Expr, vars: &HashMap<String, Kind>, abi: Abi) -> Option<Kind> {
    match e {
        Expr::Int(..) => Some(Kind::Int),
        Expr::Float(..) => Some(Kind::Float),
        Expr::Bool(..) => Some(Kind::Bool),
        Expr::Ident(n, _) => match vars.get(n) {
            Some(Kind::Poison) | None => None,
            Some(k) => Some(*k),
        },
        Expr::Neg(a, _) => match expr_kind(a, vars, abi)? {
            k @ (Kind::Int | Kind::Float) => Some(k),
            _ => None,
        },
        Expr::Not(a, _) => (expr_kind(a, vars, abi)? == Kind::Bool).then_some(Kind::Bool),
        Expr::And(a, b, _) | Expr::Or(a, b, _) => {
            (expr_kind(a, vars, abi)? == Kind::Bool && expr_kind(b, vars, abi)? == Kind::Bool).then_some(Kind::Bool)
        }
        Expr::Binary(op, a, b, _) => {
            let (ka, kb) = (expr_kind(a, vars, abi)?, expr_kind(b, vars, abi)?);
            let numeric = matches!(ka, Kind::Int | Kind::Float) && matches!(kb, Kind::Int | Kind::Float);
            match op {
                BinOp::Add | BinOp::Sub | BinOp::Mul => numeric.then(|| join(ka, kb)),
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => numeric.then_some(Kind::Bool),
                BinOp::Div => None, // int sdiv traps, float fdiv gives inf — interpreter errors on both
            }
        }
        Expr::Call(_, args, _) => {
            // Assumed same-ABI callee (the fixed point enforces it): I64 takes ints and returns int;
            // F64 takes floats (int args promoted at the call site) and returns float.
            let want = if abi == Abi::I64 { Kind::Int } else { Kind::Float };
            for a in args {
                match expr_kind(a, vars, abi)? {
                    Kind::Int => {}
                    Kind::Float if abi == Abi::F64 => {}
                    _ => return None,
                }
            }
            Some(want)
        }
        _ => None,
    }
}

fn stmt_ok(s: &Stmt) -> bool {
    match s {
        Stmt::Let { value, .. } => expr_ok(value),
        Stmt::Assign { target: Expr::Ident(..), value, .. } => expr_ok(value),
        Stmt::Return(opt, _) => opt.as_ref().map_or(true, expr_ok),
        Stmt::Expr(e) => expr_ok(e),
        Stmt::If { branches, else_body, .. } => {
            branches.iter().all(|(c, b)| expr_ok(c) && b.iter().all(stmt_ok))
                && else_body.as_ref().map_or(true, |b| b.iter().all(stmt_ok))
        }
        Stmt::While { cond, body, .. } => expr_ok(cond) && body.iter().all(stmt_ok),
        Stmt::For { iter: Expr::Range(lo, hi, _), body, .. } => expr_ok(lo) && expr_ok(hi) && body.iter().all(stmt_ok),
        Stmt::Break(_) | Stmt::Continue(_) => true,
        _ => false,
    }
}

fn expr_ok(e: &Expr) -> bool {
    match e {
        Expr::Int(..) | Expr::Float(..) | Expr::Bool(..) | Expr::Ident(..) => true,
        Expr::Neg(a, _) | Expr::Not(a, _) => expr_ok(a),
        Expr::And(a, b, _) | Expr::Or(a, b, _) => expr_ok(a) && expr_ok(b),
        Expr::Binary(op, a, b, _) => !matches!(op, BinOp::Div) && expr_ok(a) && expr_ok(b),
        Expr::Call(_, args, _) => args.iter().all(expr_ok), // target vetted by the fixed point (v0.41)
        _ => false, // strings, indexing, tensors, division, lambdas → ineligible
    }
}

/// Every `f(args)` call site in a body — (callee name, arg count).
fn collect_calls(body: &[Stmt], into: &mut Vec<(String, usize)>) {
    fn expr(e: &Expr, into: &mut Vec<(String, usize)>) {
        match e {
            Expr::Call(name, args, _) => {
                into.push((name.clone(), args.len()));
                for a in args {
                    expr(a, into);
                }
            }
            Expr::Neg(a, _) | Expr::Not(a, _) => expr(a, into),
            Expr::And(a, b, _) | Expr::Or(a, b, _) | Expr::Binary(_, a, b, _) => {
                expr(a, into);
                expr(b, into);
            }
            _ => {}
        }
    }
    for s in body {
        match s {
            Stmt::Let { value, .. } | Stmt::Assign { value, .. } => expr(value, into),
            Stmt::Return(Some(e), _) | Stmt::Expr(e) => expr(e, into),
            Stmt::If { branches, else_body, .. } => {
                for (c, b) in branches {
                    expr(c, into);
                    collect_calls(b, into);
                }
                if let Some(b) = else_body {
                    collect_calls(b, into);
                }
            }
            Stmt::While { cond, body, .. } => {
                expr(cond, into);
                collect_calls(body, into);
            }
            Stmt::For { iter: Expr::Range(lo, hi, _), body, .. } => {
                expr(lo, into);
                expr(hi, into);
                collect_calls(body, into);
            }
            _ => {}
        }
    }
}

/// Does every control path end in `return`? Falling off the end returns Unit on the tree-walker but a
/// number natively — divergent, so such functions are ineligible.
fn always_returns(body: &[Stmt]) -> bool {
    match body.last() {
        Some(Stmt::Return(..)) => true,
        Some(Stmt::If { branches, else_body, .. }) => {
            else_body.as_ref().map_or(false, |b| always_returns(b)) && branches.iter().all(|(_, b)| always_returns(b))
        }
        _ => false,
    }
}

/// A JIT-compiled function: a raw native code pointer + its arity + ABI (+ whether the i64 result
/// means a wide bool — comparisons etc. return machine 0/1 and are re-wrapped at dispatch, v0.55).
pub struct Compiled {
    ptr: *const u8,
    arity: usize,
    abi: Abi,
    ret_bool: bool,
}

impl Compiled {
    pub fn abi(&self) -> Abi {
        self.abi
    }

    pub fn ret_bool(&self) -> bool {
        self.ret_bool
    }

    /// Call an I64-ABI function with integer args (arities 0–4).
    pub fn call(&self, args: &[i64]) -> Option<i64> {
        if self.abi != Abi::I64 || args.len() != self.arity {
            return None;
        }
        unsafe {
            match self.arity {
                0 => Some(std::mem::transmute::<_, extern "C" fn() -> i64>(self.ptr)()),
                1 => Some(std::mem::transmute::<_, extern "C" fn(i64) -> i64>(self.ptr)(args[0])),
                2 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64) -> i64>(self.ptr)(args[0], args[1])),
                3 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2])),
                4 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2], args[3])),
                5 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2], args[3], args[4])),
                6 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5])),
                7 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6])),
                8 => Some(std::mem::transmute::<_, extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])),
                _ => None,
            }
        }
    }

    /// Call an F64-ABI function with float args (arities 0–4).
    pub fn call_f64(&self, args: &[f64]) -> Option<f64> {
        if self.abi != Abi::F64 || args.len() != self.arity {
            return None;
        }
        unsafe {
            match self.arity {
                0 => Some(std::mem::transmute::<_, extern "C" fn() -> f64>(self.ptr)()),
                1 => Some(std::mem::transmute::<_, extern "C" fn(f64) -> f64>(self.ptr)(args[0])),
                2 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64) -> f64>(self.ptr)(args[0], args[1])),
                3 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2])),
                4 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2], args[3])),
                5 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2], args[3], args[4])),
                6 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64, f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5])),
                7 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64, f64, f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6])),
                8 => Some(std::mem::transmute::<_, extern "C" fn(f64, f64, f64, f64, f64, f64, f64, f64) -> f64>(self.ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7])),
                _ => None,
            }
        }
    }
}

/// Owns the JIT module (keeps the executable memory alive for the program's lifetime).
pub struct Jit {
    module: JITModule,
    counter: usize,
}

impl Jit {
    pub fn new() -> Result<Self, String> {
        // opt_level=speed: without it Cranelift keeps every variable in a stack slot (~60x slower
        // loops — measured in bench/loop.wide).
        let builder = JITBuilder::with_flags(&[("opt_level", "speed")], default_libcall_names())
            .map_err(|e| format!("jit init: {}", e))?;
        Ok(Jit { module: JITModule::new(builder), counter: 0 })
    }

    /// Compile a batch of numeric functions (arities 0–4) to native code. All functions are *declared*
    /// first so bodies can call each other (direct/mutual recursion), then each is defined, then the
    /// module is finalized once. `abis` must be the classification from `eligible_set`.
    pub fn compile_batch(&mut self, funcs: &[FnDef], abis: &HashMap<String, (Abi, bool)>) -> Result<Vec<(String, Compiled)>, String> {
        let cl_ty = |abi: Abi| if abi == Abi::I64 { types::I64 } else { types::F64 };
        // Phase 1 — declare every function (so calls can reference any of them).
        let mut ids: HashMap<String, FuncId> = HashMap::new();
        for (name, params, _) in funcs {
            if params.len() > 8 {
                return Err(format!("jit: '{}' arity > 8 not supported yet", name));
            }
            let (abi, _) = abis[name.as_str()];
            let mut sig = self.module.make_signature();
            for _ in params {
                sig.params.push(AbiParam::new(cl_ty(abi)));
            }
            sig.returns.push(AbiParam::new(cl_ty(abi)));
            self.counter += 1;
            let sym = format!("__wide_jit_{}_{}", self.counter, name);
            let id = self
                .module
                .declare_function(&sym, Linkage::Export, &sig)
                .map_err(|e| format!("jit declare '{}': {}", name, e))?;
            ids.insert(name.clone(), id);
        }
        // Phase 2 — define each body (calls resolve through the declared ids).
        for (name, params, body) in funcs {
            let (abi, _) = abis[name.as_str()];
            let (kinds, _) = classify(params, body, abi).ok_or_else(|| format!("jit: '{}' failed re-analysis", name))?;
            let mut sig = self.module.make_signature();
            for _ in params {
                sig.params.push(AbiParam::new(cl_ty(abi)));
            }
            sig.returns.push(AbiParam::new(cl_ty(abi)));
            let mut ctx = self.module.make_context();
            ctx.func.signature = sig;
            let mut fbctx = FunctionBuilderContext::new();
            {
                let mut b = FunctionBuilder::new(&mut ctx.func, &mut fbctx);
                let entry = b.create_block();
                b.append_block_params_for_function_params(entry);
                b.switch_to_block(entry);

                // Declare a Cranelift variable per wide variable, typed by its kind (Float → F64).
                let mut names: Vec<String> = params.to_vec();
                collect_names(body, &mut names);
                let mut vars: HashMap<String, Variable> = HashMap::new();
                for n in &names {
                    let ty = if kinds.get(n) == Some(&Kind::Float) { types::F64 } else { types::I64 };
                    let v = b.declare_var(ty);
                    vars.insert(n.clone(), v);
                }
                // Bind parameters to their variables.
                for (i, p) in params.iter().enumerate() {
                    let pv = b.block_params(entry)[i];
                    b.def_var(vars[p], pv);
                }
                // Zero-init non-parameter variables (by type).
                for n in &names {
                    if !params.contains(n) {
                        let z = if kinds.get(n) == Some(&Kind::Float) {
                            b.ins().f64const(0.0)
                        } else {
                            b.ins().iconst(types::I64, 0)
                        };
                        b.def_var(vars[n], z);
                    }
                }

                let mut g = Gen { b, vars, loops: Vec::new(), abi, kinds: &kinds, module: &mut self.module, fn_ids: &ids };
                let terminated = g.stmts(body);
                if !terminated {
                    // unreachable: always_returns() is part of eligibility — but keep the IR valid.
                    let z = if abi == Abi::I64 { g.b.ins().iconst(types::I64, 0) } else { g.b.ins().f64const(0.0) };
                    g.b.ins().return_(&[z]);
                }
                g.b.seal_all_blocks();
                g.b.finalize();
            }
            self.module.define_function(ids[name], &mut ctx).map_err(|e| format!("jit define '{}': {}", name, e))?;
            self.module.clear_context(&mut ctx);
        }
        // Phase 3 — finalize once, hand out the native pointers.
        self.module.finalize_definitions().map_err(|e| format!("jit finalize: {}", e))?;
        let mut out = Vec::with_capacity(funcs.len());
        for (name, params, _) in funcs {
            let ptr = self.module.get_finalized_function(ids[name]);
            let (abi, ret_bool) = abis[name.as_str()];
            out.push((name.clone(), Compiled { ptr, arity: params.len(), abi, ret_bool }));
        }
        Ok(out)
    }
}

/// Collect every variable name bound in a body (let names, ident-assignment targets) — declared up front.
fn collect_names(body: &[Stmt], into: &mut Vec<String>) {
    for s in body {
        match s {
            Stmt::Let { name, .. } => push_unique(into, name),
            Stmt::Assign { target: Expr::Ident(n, _), .. } => push_unique(into, n),
            Stmt::If { branches, else_body, .. } => {
                for (_, b) in branches {
                    collect_names(b, into);
                }
                if let Some(b) = else_body {
                    collect_names(b, into);
                }
            }
            Stmt::While { body, .. } => collect_names(body, into),
            Stmt::For { var, body, .. } => {
                push_unique(into, var);
                collect_names(body, into);
            }
            _ => {}
        }
    }
}

fn push_unique(v: &mut Vec<String>, n: &str) {
    if !v.iter().any(|x| x == n) {
        v.push(n.to_string());
    }
}

struct Gen<'a> {
    b: FunctionBuilder<'a>,
    vars: HashMap<String, Variable>,
    loops: Vec<(Block, Block)>, // (header for continue, exit for break)
    abi: Abi,
    kinds: &'a HashMap<String, Kind>,    // final variable kinds (typing for vars and promotions)
    module: &'a mut JITModule,           // to make FuncRefs for calls (v0.41)
    fn_ids: &'a HashMap<String, FuncId>, // every function in the batch, callable from any body
}

impl Gen<'_> {
    /// Emit a list of statements. Returns true if control flow was terminated (a `return`/jump emitted).
    fn stmts(&mut self, body: &[Stmt]) -> bool {
        for s in body {
            if self.stmt(s) {
                return true;
            }
        }
        false
    }

    fn stmt(&mut self, s: &Stmt) -> bool {
        match s {
            Stmt::Let { name, value, .. } | Stmt::Assign { target: Expr::Ident(name, _), value, .. } => {
                let (v, k) = self.expr(value);
                let v = if self.kinds.get(name) == Some(&Kind::Float) { self.promote(v, k) } else { v };
                let var = self.vars[name];
                self.b.def_var(var, v);
                false
            }
            Stmt::Expr(e) => {
                self.expr(e);
                false
            }
            Stmt::Return(opt, _) => {
                let v = match opt {
                    Some(e) => {
                        let (v, k) = self.expr(e);
                        if self.abi == Abi::F64 {
                            self.promote(v, k)
                        } else {
                            v
                        }
                    }
                    None => self.b.ins().iconst(types::I64, 0), // unreachable (eligibility)
                };
                self.b.ins().return_(&[v]);
                true
            }
            Stmt::If { branches, else_body, .. } => self.gen_if(branches, else_body),
            Stmt::While { cond, body, .. } => self.gen_while(cond, body),
            Stmt::For { var, iter: Expr::Range(lo, hi, _), body, .. } => self.gen_for_range(var, lo, hi, body),
            Stmt::Break(_) => {
                let exit = self.loops.last().unwrap().1;
                self.b.ins().jump(exit, &[]);
                true
            }
            Stmt::Continue(_) => {
                let header = self.loops.last().unwrap().0;
                self.b.ins().jump(header, &[]);
                true
            }
            _ => false, // unreachable: eligibility guaranteed it
        }
    }

    fn gen_if(&mut self, branches: &[(Expr, Vec<Stmt>)], else_body: &Option<Vec<Stmt>>) -> bool {
        let merge = self.b.create_block();
        // Lower `if c1 {..} elif c2 {..} else {..}` as a chain of brif.
        for (cond, body) in branches {
            let (cv, _) = self.expr(cond); // Bool kind → i64 0/1
            let then_blk = self.b.create_block();
            let next_blk = self.b.create_block();
            self.b.ins().brif(cv, then_blk, &[], next_blk, &[]);
            self.b.switch_to_block(then_blk);
            let term = self.stmts(body);
            if !term {
                self.b.ins().jump(merge, &[]);
            }
            self.b.switch_to_block(next_blk);
        }
        match else_body {
            Some(body) => {
                let term = self.stmts(body);
                if !term {
                    self.b.ins().jump(merge, &[]);
                }
            }
            None => {
                self.b.ins().jump(merge, &[]);
            }
        }
        self.b.switch_to_block(merge);
        false
    }

    fn gen_while(&mut self, cond: &Expr, body: &[Stmt]) -> bool {
        let header = self.b.create_block();
        let body_blk = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let (cv, _) = self.expr(cond);
        self.b.ins().brif(cv, body_blk, &[], exit, &[]);
        self.b.switch_to_block(body_blk);
        self.loops.push((header, exit));
        let term = self.stmts(body);
        if !term {
            self.b.ins().jump(header, &[]); // loop back
        }
        self.loops.pop();
        self.b.switch_to_block(exit);
        false
    }

    /// `for i in lo..hi { ... }` over integers (v0.55). The range bounds evaluate once; `continue`
    /// jumps to the *increment* block (a plain header jump would skip `i += 1` and loop forever).
    fn gen_for_range(&mut self, var: &str, lo: &Expr, hi: &Expr, body: &[Stmt]) -> bool {
        let (lo_v, _) = self.expr(lo);
        let (hi_v, _) = self.expr(hi);
        let end_var = self.b.declare_var(types::I64);
        self.b.def_var(end_var, hi_v);
        let iv = self.vars[var];
        self.b.def_var(iv, lo_v);
        let header = self.b.create_block();
        let body_blk = self.b.create_block();
        let incr = self.b.create_block();
        let exit = self.b.create_block();
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(header);
        let cur = self.b.use_var(iv);
        let end = self.b.use_var(end_var);
        let c = self.b.ins().icmp(IntCC::SignedLessThan, cur, end);
        self.b.ins().brif(c, body_blk, &[], exit, &[]);
        self.b.switch_to_block(body_blk);
        self.loops.push((incr, exit));
        let term = self.stmts(body);
        if !term {
            self.b.ins().jump(incr, &[]);
        }
        self.loops.pop();
        self.b.switch_to_block(incr);
        let cur = self.b.use_var(iv);
        let one = self.b.ins().iconst(types::I64, 1);
        let next = self.b.ins().iadd(cur, one);
        self.b.def_var(iv, next);
        self.b.ins().jump(header, &[]);
        self.b.switch_to_block(exit);
        false
    }

    /// Int-kind value → f64 (numeric promotion, mirrors the tree-walker's int→float promotion).
    fn promote(&mut self, v: Value, k: Kind) -> Value {
        if k == Kind::Int {
            self.b.ins().fcvt_from_sint(types::F64, v)
        } else {
            v
        }
    }

    /// Lower an expression; returns the value and its kind (bools are i64 0/1).
    fn expr(&mut self, e: &Expr) -> (Value, Kind) {
        match e {
            Expr::Int(n, _) => (self.b.ins().iconst(types::I64, *n), Kind::Int),
            Expr::Float(x, _) => (self.b.ins().f64const(*x), Kind::Float),
            Expr::Bool(bv, _) => (self.b.ins().iconst(types::I64, if *bv { 1 } else { 0 }), Kind::Bool),
            Expr::Ident(n, _) => {
                let k = *self.kinds.get(n).unwrap_or(&Kind::Int);
                (self.b.use_var(self.vars[n]), k)
            }
            Expr::Neg(a, _) => {
                let (v, k) = self.expr(a);
                match k {
                    Kind::Float => (self.b.ins().fneg(v), Kind::Float),
                    _ => (self.b.ins().ineg(v), Kind::Int),
                }
            }
            Expr::Not(a, _) => {
                let (v, _) = self.expr(a);
                let one = self.b.ins().iconst(types::I64, 1);
                (self.b.ins().bxor(v, one), Kind::Bool) // v∈{0,1} → flip
            }
            Expr::And(a, b, _) => {
                let (x, _) = self.expr(a);
                let (y, _) = self.expr(b);
                (self.b.ins().band(x, y), Kind::Bool)
            }
            Expr::Or(a, b, _) => {
                let (x, _) = self.expr(a);
                let (y, _) = self.expr(b);
                (self.b.ins().bor(x, y), Kind::Bool)
            }
            Expr::Call(name, args, _) => {
                // Native-to-native call (v0.41): same-ABI callee (fixed point); F64 promotes int args.
                let want_float = self.abi == Abi::F64;
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    let (v, k) = self.expr(a);
                    vals.push(if want_float { self.promote(v, k) } else { v });
                }
                let fref = self.module.declare_func_in_func(self.fn_ids[name.as_str()], self.b.func);
                let call = self.b.ins().call(fref, &vals);
                let res = self.b.inst_results(call)[0];
                (res, if want_float { Kind::Float } else { Kind::Int })
            }
            Expr::Binary(op, a, b, _) => {
                let (mut x, kx) = self.expr(a);
                let (mut y, ky) = self.expr(b);
                let float = kx == Kind::Float || ky == Kind::Float;
                if float {
                    x = self.promote(x, kx);
                    y = self.promote(y, ky);
                }
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul => {
                        let v = if float {
                            match op {
                                BinOp::Add => self.b.ins().fadd(x, y),
                                BinOp::Sub => self.b.ins().fsub(x, y),
                                _ => self.b.ins().fmul(x, y),
                            }
                        } else {
                            match op {
                                BinOp::Add => self.b.ins().iadd(x, y),
                                BinOp::Sub => self.b.ins().isub(x, y),
                                _ => self.b.ins().imul(x, y),
                            }
                        };
                        (v, if float { Kind::Float } else { Kind::Int })
                    }
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge => {
                        let c = if float {
                            let cc = match op {
                                BinOp::Eq => FloatCC::Equal,
                                BinOp::Ne => FloatCC::NotEqual,
                                BinOp::Lt => FloatCC::LessThan,
                                BinOp::Gt => FloatCC::GreaterThan,
                                BinOp::Le => FloatCC::LessThanOrEqual,
                                BinOp::Ge => FloatCC::GreaterThanOrEqual,
                                _ => unreachable!(),
                            };
                            self.b.ins().fcmp(cc, x, y)
                        } else {
                            let cc = match op {
                                BinOp::Eq => IntCC::Equal,
                                BinOp::Ne => IntCC::NotEqual,
                                BinOp::Lt => IntCC::SignedLessThan,
                                BinOp::Gt => IntCC::SignedGreaterThan,
                                BinOp::Le => IntCC::SignedLessThanOrEqual,
                                BinOp::Ge => IntCC::SignedGreaterThanOrEqual,
                                _ => unreachable!(),
                            };
                            self.b.ins().icmp(cc, x, y)
                        };
                        (self.b.ins().uextend(types::I64, c), Kind::Bool)
                    }
                    BinOp::Div => unreachable!("division is ineligible for the JIT"),
                }
            }
            _ => unreachable!("eligibility guaranteed a numeric expression"),
        }
    }
}

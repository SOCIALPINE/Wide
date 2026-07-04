//! AST → bytecode compiler (stage 2). Turns the tree into a flat instruction stream once, up front —
//! that's where the "compiler advantages" live (the work happens at compile time, not per-execution).
//!
//! v0.19 covers the core subset. Anything not yet handled returns a clear error naming the construct,
//! so the tree-walker (which handles everything) stays the fallback. Block-local scoping for loop/if
//! bodies is not modeled yet (bodies run in the enclosing scope) — refined when slot allocation lands.

use crate::ast::*;
use crate::bytecode::*;
use crate::span::Span;
use crate::value::Value;

/// Compile a function/method body into a chunk (falls off the end → returns Unit).
fn compile_fn_body(body: &[Stmt], span: Span) -> Result<Chunk, String> {
    let mut c = Compiler::new();
    c.stmts(body)?;
    c.chunk.emit(Op::Unit, span);
    c.chunk.emit(Op::Return, span);
    Ok(c.chunk)
}

/// Compile a whole program: hoist functions/structs/enums/impls, then compile the top level into `main`.
pub fn compile(prog: &[Stmt]) -> Result<Program, String> {
    let mut p = Program::default();
    for s in prog {
        match s {
            Stmt::Fn { name, params, body, span, .. } => {
                p.funcs.insert(name.clone(), CompiledFn { params: params.clone(), chunk: compile_fn_body(body, *span)? });
            }
            Stmt::Struct { name, fields, .. } => {
                p.structs.insert(name.clone(), fields.clone());
            }
            Stmt::Enum { name, variants, .. } => {
                p.enums.insert(name.clone(), variants.iter().cloned().collect());
            }
            Stmt::Impl { type_name, methods, .. } => {
                let table = p.methods.entry(type_name.clone()).or_default();
                for m in methods {
                    if let Stmt::Fn { name, params, body, span, .. } = m {
                        table.insert(name.clone(), CompiledFn { params: params.clone(), chunk: compile_fn_body(body, *span)? });
                    }
                }
            }
            // std/ imports are built-in markers (no codegen). File imports are removed by load_file
            // before compiling, so a non-std import here means compile() was called on a raw file.
            Stmt::Import(p, span) => {
                if !p.starts_with("std/") {
                    return Err(format!("line {}: file imports must go through load_file (the VM CLI does this)", span.line));
                }
            }
            _ => {}
        }
    }
    let mut main = Compiler::new();
    main.stmts(prog)?;
    p.main = main.chunk;
    Ok(p)
}

struct LoopCtx {
    breaks: Vec<usize>,    // placeholder Jump indices → patch to loop exit
    continues: Vec<usize>, // placeholder Jump indices → patch to loop continue target
}

struct Compiler {
    chunk: Chunk,
    loops: Vec<LoopCtx>,
    hidden: usize, // counter for synthetic internal variable names ($end0, ...)
}

impl Compiler {
    fn new() -> Self {
        Compiler { chunk: Chunk::new(), loops: Vec::new(), hidden: 0 }
    }

    fn here(&self) -> usize {
        self.chunk.code.len()
    }

    /// Backpatch a jump instruction's target to `target`.
    fn patch(&mut self, at: usize, target: usize) {
        self.chunk.code[at] = match &self.chunk.code[at] {
            Op::Jump(_) => Op::Jump(target),
            Op::JumpIfFalse(_) => Op::JumpIfFalse(target),
            Op::JumpIfFalsePeek(_) => Op::JumpIfFalsePeek(target),
            Op::JumpIfTruePeek(_) => Op::JumpIfTruePeek(target),
            Op::TryMatch(pi, _) => Op::TryMatch(*pi, target),
            other => other.clone(),
        };
    }

    fn hidden_name(&mut self) -> String {
        let n = format!("$end{}", self.hidden);
        self.hidden += 1;
        n
    }

    fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for s in stmts {
            self.stmt(s)?;
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<(), String> {
        match s {
            Stmt::Fn { .. } => {} // hoisted separately
            Stmt::Let { name, ty, value, span } => {
                self.expr(value)?;
                let i = self.chunk.add_name(name);
                // Typed `let x: T = e` defines in the current scope; bare `x = e` is Stmt::Assign.
                let op = if ty.is_some() { Op::DefineVar(i) } else { Op::AssignVar(i) };
                self.chunk.emit(op, *span);
            }
            Stmt::Assign { target, value, span } => match target {
                Expr::Ident(name, _) => {
                    self.expr(value)?;
                    let i = self.chunk.add_name(name);
                    self.chunk.emit(Op::AssignVar(i), *span);
                }
                Expr::Index(recv, idx, _) => {
                    self.expr(recv)?;
                    self.expr(idx)?;
                    self.expr(value)?;
                    self.chunk.emit(Op::SetIndex, *span); // pops val, idx, recv
                }
                Expr::Field(recv, fname, _) => {
                    self.expr(recv)?;
                    self.expr(value)?;
                    let i = self.chunk.add_name(fname);
                    self.chunk.emit(Op::SetField(i), *span); // pops val, recv
                }
                Expr::Deref(inner, _) => {
                    self.expr(inner)?; // the pointer
                    self.expr(value)?;
                    self.chunk.emit(Op::DerefSet, *span); // pops val, ptr
                }
                _ => return Err(format!("line {}: this assignment target is not supported by the VM yet", span.line)),
            },
            Stmt::Expr(e) => {
                self.expr(e)?;
                self.chunk.emit(Op::Pop, expr_span(e));
            }
            Stmt::Return(opt, span) => {
                match opt {
                    Some(e) => self.expr(e)?,
                    None => {
                        self.chunk.emit(Op::Unit, *span);
                    }
                }
                self.chunk.emit(Op::Return, *span);
            }
            Stmt::If { branches, else_body, .. } => self.compile_if(branches, else_body)?,
            Stmt::While { cond, body, span } => self.compile_while(cond, body, *span)?,
            Stmt::For { var, iter, body, span } => self.compile_for(var, iter, body, *span)?,
            Stmt::Break(span) => {
                if self.loops.is_empty() {
                    return Err(format!("line {}: cannot break outside a loop", span.line));
                }
                let at = self.chunk.emit(Op::Jump(0), *span);
                self.loops.last_mut().unwrap().breaks.push(at);
            }
            Stmt::Continue(span) => {
                if self.loops.is_empty() {
                    return Err(format!("line {}: cannot continue outside a loop", span.line));
                }
                let at = self.chunk.emit(Op::Jump(0), *span);
                self.loops.last_mut().unwrap().continues.push(at);
            }
            Stmt::Struct { .. } | Stmt::Enum { .. } | Stmt::Impl { .. } => {} // hoisted separately
            Stmt::Match { subject, arms, span } => self.compile_match_stmt(subject, arms, *span)?,
            Stmt::Import(p, span) => {
                if !p.starts_with("std/") {
                    return Err(format!("line {}: file imports must go through load_file", span.line));
                }
            }
            Stmt::Cout(parts, span) => self.compile_cout(parts, *span)?,
            Stmt::Cin(targets, span) => self.compile_cin(targets, *span)?,
            Stmt::ShowProvenance(e, span) => {
                self.expr(e)?;
                self.chunk.emit(Op::ShowProvenance, *span);
            }
            Stmt::Trust(_, span) => {
                return Err(format!("line {}: @trust is not supported by the VM yet (tree-walker runs it)", span.line));
            }
        }
        Ok(())
    }

    fn compile_if(&mut self, branches: &[(Expr, Vec<Stmt>)], else_body: &Option<Vec<Stmt>>) -> Result<(), String> {
        let mut end_jumps = Vec::new();
        for (cond, body) in branches {
            self.expr(cond)?;
            let skip = self.chunk.emit(Op::JumpIfFalse(0), expr_span(cond));
            self.stmts(body)?;
            let to_end = self.chunk.emit(Op::Jump(0), expr_span(cond));
            end_jumps.push(to_end);
            let here = self.here();
            self.patch(skip, here);
        }
        if let Some(body) = else_body {
            self.stmts(body)?;
        }
        let end = self.here();
        for j in end_jumps {
            self.patch(j, end);
        }
        Ok(())
    }

    fn compile_while(&mut self, cond: &Expr, body: &[Stmt], _span: Span) -> Result<(), String> {
        let start = self.here();
        self.expr(cond)?;
        let exit = self.chunk.emit(Op::JumpIfFalse(0), expr_span(cond));
        self.loops.push(LoopCtx { breaks: Vec::new(), continues: Vec::new() });
        self.stmts(body)?;
        self.chunk.emit(Op::Jump(start), expr_span(cond));
        let ctx = self.loops.pop().unwrap();
        let exit_addr = self.here();
        self.patch(exit, exit_addr);
        for b in ctx.breaks {
            self.patch(b, exit_addr);
        }
        for c in ctx.continues {
            self.patch(c, start); // continue → re-evaluate the condition
        }
        Ok(())
    }

    /// for v in iter { body } — integer ranges, or arrays/strings (desugared to an index loop).
    fn compile_for(&mut self, var: &str, iter: &Expr, body: &[Stmt], span: Span) -> Result<(), String> {
        let (lo, hi) = match iter {
            Expr::Range(lo, hi, _) => (lo, hi),
            _ => return self.compile_for_indexed(var, iter, body, span),
        };
        let end_name = self.hidden_name();
        let vi = self.chunk.add_name(var);
        let ei = self.chunk.add_name(&end_name);
        // $end = hi ; v = lo
        self.expr(hi)?;
        self.chunk.emit(Op::DefineVar(ei), span);
        self.expr(lo)?;
        self.chunk.emit(Op::DefineVar(vi), span);
        let start = self.here();
        self.chunk.emit(Op::LoadVar(vi), span);
        self.chunk.emit(Op::LoadVar(ei), span);
        self.chunk.emit(Op::Lt, span);
        let exit = self.chunk.emit(Op::JumpIfFalse(0), span);
        self.loops.push(LoopCtx { breaks: Vec::new(), continues: Vec::new() });
        self.stmts(body)?;
        // continue target: increment v then loop
        let cont = self.here();
        self.chunk.emit(Op::LoadVar(vi), span);
        let one = self.chunk.add_const(Value::Int(1));
        self.chunk.emit(Op::Const(one), span);
        self.chunk.emit(Op::Add, span);
        self.chunk.emit(Op::AssignVar(vi), span);
        self.chunk.emit(Op::Jump(start), span);
        let ctx = self.loops.pop().unwrap();
        let exit_addr = self.here();
        self.patch(exit, exit_addr);
        for b in ctx.breaks {
            self.patch(b, exit_addr);
        }
        for c in ctx.continues {
            self.patch(c, cont);
        }
        Ok(())
    }

    /// for v in <array|string> { body } — desugar to an index loop using Len + Index.
    fn compile_for_indexed(&mut self, var: &str, iter: &Expr, body: &[Stmt], span: Span) -> Result<(), String> {
        let arr_name = self.hidden_name();
        let idx_name = self.hidden_name();
        let ai = self.chunk.add_name(&arr_name);
        let ii = self.chunk.add_name(&idx_name);
        let vi = self.chunk.add_name(var);
        // $arr = iter ; $i = 0
        self.expr(iter)?;
        self.chunk.emit(Op::DefineVar(ai), span);
        // Borrow gradient (§3.3): hold a shared borrow of the iterated buffer for the loop body
        // (no-op for non-arrays like strings). Released at every exit path below (normal + break).
        let origin = match iter {
            Expr::Ident(n, _) => n.clone(),
            _ => "array".to_string(),
        };
        let oi = self.chunk.add_name(&origin);
        self.chunk.emit(Op::LoadVar(ai), span);
        self.chunk.emit(Op::BorrowShared(oi), span);
        let zero = self.chunk.add_const(Value::Int(0));
        self.chunk.emit(Op::Const(zero), span);
        self.chunk.emit(Op::DefineVar(ii), span);
        let start = self.here();
        // $i < len($arr) ?
        self.chunk.emit(Op::LoadVar(ii), span);
        self.chunk.emit(Op::LoadVar(ai), span);
        self.chunk.emit(Op::Len, span);
        self.chunk.emit(Op::Lt, span);
        let exit = self.chunk.emit(Op::JumpIfFalse(0), span);
        // v = $arr[$i]
        self.chunk.emit(Op::LoadVar(ai), span);
        self.chunk.emit(Op::LoadVar(ii), span);
        self.chunk.emit(Op::Index, span);
        self.chunk.emit(Op::DefineVar(vi), span);
        self.loops.push(LoopCtx { breaks: Vec::new(), continues: Vec::new() });
        self.stmts(body)?;
        let cont = self.here();
        self.chunk.emit(Op::LoadVar(ii), span);
        let one = self.chunk.add_const(Value::Int(1));
        self.chunk.emit(Op::Const(one), span);
        self.chunk.emit(Op::Add, span);
        self.chunk.emit(Op::AssignVar(ii), span);
        self.chunk.emit(Op::Jump(start), span);
        let ctx = self.loops.pop().unwrap();
        // Release the shared borrow — both normal exit and `break` land here.
        let release_addr = self.here();
        self.chunk.emit(Op::LoadVar(ai), span);
        self.chunk.emit(Op::BorrowRelease, span);
        self.patch(exit, release_addr);
        for b in ctx.breaks {
            self.patch(b, release_addr);
        }
        for c in ctx.continues {
            self.patch(c, cont);
        }
        Ok(())
    }

    /// cout << a << b ... — push each value, then a single Cout that prints them concatenated.
    fn compile_cout(&mut self, parts: &[Expr], span: Span) -> Result<(), String> {
        for e in parts {
            self.expr(e)?;
        }
        self.chunk.emit(Op::Cout(parts.len()), span);
        Ok(())
    }

    /// cin >> lv1 >> lv2 ... — for each lvalue: read a token (ReadInput), then assign to it.
    fn compile_cin(&mut self, targets: &[Expr], span: Span) -> Result<(), String> {
        for t in targets {
            match t {
                Expr::Ident(name, _) => {
                    self.chunk.emit(Op::ReadInput, span);
                    let i = self.chunk.add_name(name);
                    self.chunk.emit(Op::AssignVar(i), span);
                }
                Expr::Index(recv, idx, _) => {
                    self.expr(recv)?;
                    self.expr(idx)?;
                    self.chunk.emit(Op::ReadInput, span);
                    self.chunk.emit(Op::SetIndex, span);
                }
                Expr::Field(recv, fname, _) => {
                    self.expr(recv)?;
                    self.chunk.emit(Op::ReadInput, span);
                    let i = self.chunk.add_name(fname);
                    self.chunk.emit(Op::SetField(i), span);
                }
                _ => return Err(format!("line {}: cin target must be a variable/index/field", span.line)),
            }
        }
        Ok(())
    }

    /// Statement match: run the first matching arm's body. Subject stored in a hidden var so each arm
    /// reloads it (TryMatch pops it). Non-exhaustive statement match falls through (like the tree-walker).
    fn compile_match_stmt(&mut self, subject: &Expr, arms: &[Arm], span: Span) -> Result<(), String> {
        self.expr(subject)?;
        let subj = self.hidden_name();
        let si = self.chunk.add_name(&subj);
        self.chunk.emit(Op::DefineVar(si), span);
        let mut end_jumps = Vec::new();
        for arm in arms {
            self.chunk.emit(Op::LoadVar(si), span);
            let pi = self.chunk.add_pattern(arm.pattern.clone());
            let fail = self.chunk.emit(Op::TryMatch(pi, 0), span);
            self.stmts(&arm.body)?;
            end_jumps.push(self.chunk.emit(Op::Jump(0), span));
            let here = self.here();
            self.patch(fail, here);
        }
        let end = self.here();
        for j in end_jumps {
            self.patch(j, end);
        }
        Ok(())
    }

    /// Match expression: yields the matching arm's value. Non-exhaustive → MatchFail (runtime error).
    fn compile_match_expr(&mut self, subject: &Expr, arms: &[(Pattern, Expr)], span: Span) -> Result<(), String> {
        self.expr(subject)?;
        let subj = self.hidden_name();
        let si = self.chunk.add_name(&subj);
        self.chunk.emit(Op::DefineVar(si), span);
        let mut end_jumps = Vec::new();
        for (pat, body) in arms {
            self.chunk.emit(Op::LoadVar(si), span);
            let pi = self.chunk.add_pattern(pat.clone());
            let fail = self.chunk.emit(Op::TryMatch(pi, 0), span);
            self.expr(body)?;
            end_jumps.push(self.chunk.emit(Op::Jump(0), span));
            let here = self.here();
            self.patch(fail, here);
        }
        self.chunk.emit(Op::MatchFail, span);
        let end = self.here();
        for j in end_jumps {
            self.patch(j, end);
        }
        Ok(())
    }

    fn expr(&mut self, e: &Expr) -> Result<(), String> {
        match e {
            Expr::Int(n, span) => {
                let i = self.chunk.add_const(Value::Int(*n));
                self.chunk.emit(Op::Const(i), *span);
            }
            Expr::Float(x, span) => {
                let i = self.chunk.add_const(Value::Float(*x));
                self.chunk.emit(Op::Const(i), *span);
            }
            Expr::Str(s, span) => {
                let i = self.chunk.add_const(Value::Str(s.clone()));
                self.chunk.emit(Op::Const(i), *span);
            }
            Expr::Bool(b, span) => {
                self.chunk.emit(if *b { Op::True } else { Op::False }, *span);
            }
            Expr::Ident(name, span) => {
                let i = self.chunk.add_name(name);
                self.chunk.emit(Op::LoadVar(i), *span);
            }
            Expr::Neg(inner, span) => {
                self.expr(inner)?;
                self.chunk.emit(Op::Neg, *span);
            }
            Expr::Not(inner, span) => {
                self.expr(inner)?;
                self.chunk.emit(Op::Not, *span);
            }
            Expr::And(l, r, span) => {
                self.expr(l)?;
                let end = self.chunk.emit(Op::JumpIfFalsePeek(0), *span);
                self.chunk.emit(Op::Pop, *span);
                self.expr(r)?;
                self.chunk.emit(Op::BoolCheck, *span);
                let here = self.here();
                self.patch(end, here);
            }
            Expr::Or(l, r, span) => {
                self.expr(l)?;
                let end = self.chunk.emit(Op::JumpIfTruePeek(0), *span);
                self.chunk.emit(Op::Pop, *span);
                self.expr(r)?;
                self.chunk.emit(Op::BoolCheck, *span);
                let here = self.here();
                self.patch(end, here);
            }
            Expr::Binary(op, l, r, span) => {
                self.expr(l)?;
                self.expr(r)?;
                let o = match op {
                    BinOp::Add => Op::Add,
                    BinOp::Sub => Op::Sub,
                    BinOp::Mul => Op::Mul,
                    BinOp::Div => Op::Div,
                    BinOp::Eq => Op::Eq,
                    BinOp::Ne => Op::Ne,
                    BinOp::Lt => Op::Lt,
                    BinOp::Gt => Op::Gt,
                    BinOp::Le => Op::Le,
                    BinOp::Ge => Op::Ge,
                };
                self.chunk.emit(o, *span);
            }
            Expr::Call(name, args, span) => {
                for a in args {
                    self.expr(a)?;
                }
                if name == "print" {
                    self.chunk.emit(Op::Print(args.len()), *span);
                } else {
                    let i = self.chunk.add_name(name);
                    self.chunk.emit(Op::Call(i, args.len()), *span);
                }
            }
            Expr::Array(elems, span) => {
                for el in elems {
                    self.expr(el)?;
                }
                self.chunk.emit(Op::Array(elems.len()), *span);
            }
            Expr::Index(recv, idx, span) => {
                self.expr(recv)?;
                self.expr(idx)?;
                self.chunk.emit(Op::Index, *span);
            }
            Expr::Method(recv, name, args, span) => {
                // raw.* soft namespace (memory model §3.2) — compile to RawOp, mirroring the tree-walker.
                if matches!(recv.as_ref(), Expr::Ident(n, _) if n == "raw") {
                    for a in args {
                        self.expr(a)?;
                    }
                    let i = self.chunk.add_name(name);
                    self.chunk.emit(Op::RawOp(i, args.len()), *span);
                    return Ok(());
                }
                self.expr(recv)?;
                for a in args {
                    self.expr(a)?;
                }
                let i = self.chunk.add_name(name);
                self.chunk.emit(Op::Method(i, args.len()), *span);
            }
            Expr::Field(recv, name, span) => {
                self.expr(recv)?;
                let i = self.chunk.add_name(name);
                self.chunk.emit(Op::Field(i), *span);
            }
            Expr::Map(span) => {
                self.chunk.emit(Op::MapNew, *span);
            }
            Expr::Range(lo, hi, span) => {
                self.expr(lo)?;
                self.expr(hi)?;
                self.chunk.emit(Op::Range, *span);
            }
            Expr::Try(inner, span) => {
                self.expr(inner)?;
                self.chunk.emit(Op::Try, *span);
            }
            Expr::StructLit(name, fields, span) => {
                let mut fidx = Vec::with_capacity(fields.len());
                for (fname, fexpr) in fields {
                    self.expr(fexpr)?;
                    fidx.push(self.chunk.add_name(fname));
                }
                let ni = self.chunk.add_name(name);
                self.chunk.emit(Op::StructLit(ni, fidx), *span);
            }
            Expr::EnumLit(ename, variant, args, span) => {
                for a in args {
                    self.expr(a)?;
                }
                let ei = self.chunk.add_name(ename);
                let vi = self.chunk.add_name(variant);
                self.chunk.emit(Op::EnumLit(ei, vi, args.len()), *span);
            }
            Expr::Match(subject, arms, span) => self.compile_match_expr(subject, arms, *span)?,
            Expr::AddrOf(inner, span) => {
                // &xs[i] — only array-element addresses (memory model §3.1), mirroring the tree-walker.
                match inner.as_ref() {
                    Expr::Index(arr_e, idx_e, _) => {
                        self.expr(arr_e)?;
                        self.expr(idx_e)?;
                        let name = match arr_e.as_ref() {
                            Expr::Ident(n, _) => n.clone(),
                            _ => "?".to_string(),
                        };
                        let i = self.chunk.add_name(&name);
                        self.chunk.emit(Op::AddrOf(i), *span);
                    }
                    _ => return Err(format!("line {}: can only take & of an array element yet (e.g. &xs[i])", span.line)),
                }
            }
            Expr::Deref(inner, span) => {
                self.expr(inner)?;
                self.chunk.emit(Op::Deref, *span);
            }
            Expr::Lambda(_, _, span) | Expr::CallValue(_, _, span) => {
                return Err(format!("line {}: closures / function values are not supported by the VM yet (tree-walker runs them)", span.line));
            }
            Expr::AddrOfMut(_, span) => {
                return Err(format!("line {}: &mut borrows are not supported by the VM yet (tree-walker runs them)", span.line));
            }
        }
        Ok(())
    }
}

/// Best-effort span of an expression (for `Pop`/diagnostics; also used by the borrow static pass).
pub fn expr_span(e: &Expr) -> Span {
    match e {
        Expr::Int(_, s) | Expr::Float(_, s) | Expr::Bool(_, s) | Expr::Str(_, s) | Expr::Ident(_, s)
        | Expr::Array(_, s) | Expr::Map(s) | Expr::Range(_, _, s) | Expr::Neg(_, s) | Expr::Not(_, s)
        | Expr::And(_, _, s) | Expr::Or(_, _, s) | Expr::Binary(_, _, _, s) | Expr::Call(_, _, s)
        | Expr::Method(_, _, _, s) | Expr::Field(_, _, s) | Expr::Index(_, _, s) | Expr::StructLit(_, _, s)
        | Expr::EnumLit(_, _, _, s) | Expr::Match(_, _, s) | Expr::Try(_, s)
        | Expr::AddrOf(_, s) | Expr::AddrOfMut(_, s) | Expr::Deref(_, s) | Expr::Lambda(_, _, s) | Expr::CallValue(_, _, s) => *s,
    }
}

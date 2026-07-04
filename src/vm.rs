//! Stack-based VM that executes bytecode (stage 2). The tree-walker (eval.rs) stays the reference;
//! this runs the compiled `Program` for the supported core subset. Values are the same dynamic `Value`
//! as the tree-walker, so semantics match — speed (slot allocation, JIT specialization) comes later.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::rc::Rc;

use crate::ast::BinOp;
use crate::bytecode::{Chunk, CompiledFn, Op, Program};
use crate::eval::{match_pattern, parse_input_token};
use crate::lumen::Channel;
use crate::runtime;
use crate::value::{value_cmp, MapKey, Value};

// Operators + core builtins are shared with the tree-walker via `runtime` (one source of truth).
// Still local (tech debt, to migrate to `runtime` next): array/map/str method + index dispatch.

pub struct Vm {
    globals: HashMap<String, Value>,
    funcs: HashMap<String, Rc<CompiledFn>>,
    structs: HashMap<String, Vec<String>>,
    enums: HashMap<String, HashMap<String, usize>>,
    methods: HashMap<String, HashMap<String, Rc<CompiledFn>>>,
    input: VecDeque<String>, // pending whitespace tokens for `cin`
    stdin_enabled: bool,     // refill from real stdin when input is empty (false in tests)
    borrows: runtime::Borrows, // active shared borrows on array buffers (memory model §3.3, shared with eval)
    pub channel: Channel,
}

impl Default for Vm {
    fn default() -> Self {
        Self::new()
    }
}

impl Vm {
    pub fn new() -> Self {
        Vm {
            globals: HashMap::new(),
            funcs: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            methods: HashMap::new(),
            input: VecDeque::new(),
            stdin_enabled: true,
            borrows: HashMap::new(),
            channel: Channel::new(),
        }
    }

    pub fn get(&self, name: &str) -> Option<Value> {
        self.globals.get(name).cloned()
    }

    /// Feed `cin` from a string instead of real stdin (for tests / embedding).
    pub fn set_input(&mut self, s: &str) {
        self.input = s.split_whitespace().map(|t| t.to_string()).collect();
        self.stdin_enabled = false;
    }

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
                Ok(0) | Err(_) => return None,
                Ok(_) => {
                    for tok in line.split_whitespace() {
                        self.input.push_back(tok.to_string());
                    }
                }
            }
        }
    }

    pub fn run(&mut self, prog: &Program) -> Result<(), String> {
        for (k, v) in &prog.funcs {
            self.funcs.insert(k.clone(), Rc::new(v.clone()));
        }
        self.structs = prog.structs.clone();
        self.enums = prog.enums.clone();
        for (sname, ms) in &prog.methods {
            let table = self.methods.entry(sname.clone()).or_default();
            for (mname, f) in ms {
                table.insert(mname.clone(), Rc::new(f.clone()));
            }
        }
        let mut scopes: Vec<HashMap<String, Value>> = Vec::new();
        match self.run_chunk(&prog.main, &mut scopes)? {
            // A top-level `?` that propagated an error surfaces as a program error.
            Some(Value::Err(e)) => Err(format!("error propagated at top level: {}", e)),
            _ => Ok(()),
        }
    }

    // ---- variable resolution (mirrors the tree-walker's scope chain) ----

    fn load(&self, name: &str, scopes: &[HashMap<String, Value>], line: usize) -> Result<Value, String> {
        for s in scopes.iter().rev() {
            if let Some(v) = s.get(name) {
                return Ok(v.clone());
            }
        }
        self.globals
            .get(name)
            .cloned()
            .ok_or_else(|| format!("line {}: undefined name '{}'", line, name))
    }

    fn define(&mut self, name: &str, v: Value, scopes: &mut [HashMap<String, Value>]) {
        if let Some(top) = scopes.last_mut() {
            top.insert(name.to_string(), v);
        } else {
            self.globals.insert(name.to_string(), v);
        }
    }

    fn assign(&mut self, name: &str, v: Value, scopes: &mut [HashMap<String, Value>]) {
        for s in scopes.iter_mut().rev() {
            if s.contains_key(name) {
                s.insert(name.to_string(), v);
                return;
            }
        }
        if self.globals.contains_key(name) {
            self.globals.insert(name.to_string(), v);
            return;
        }
        self.define(name, v, scopes);
    }

    /// Execute one chunk. Returns Some(value) if a `Return` ran, None if it fell off the end.
    fn run_chunk(&mut self, chunk: &Chunk, scopes: &mut Vec<HashMap<String, Value>>) -> Result<Option<Value>, String> {
        let mut stack: Vec<Value> = Vec::new();
        let mut ip = 0usize;
        while ip < chunk.code.len() {
            let span = chunk.spans[ip];
            let line = span.line;
            match &chunk.code[ip] {
                Op::Const(i) => stack.push(chunk.consts[*i].clone()),
                Op::True => stack.push(Value::Bool(true)),
                Op::False => stack.push(Value::Bool(false)),
                Op::Unit => stack.push(Value::Unit),
                Op::Pop => {
                    stack.pop();
                }
                Op::Neg => {
                    let v = pop(&mut stack);
                    stack.push(match v {
                        Value::Int(n) => Value::Int(-n),
                        Value::Float(x) => Value::Float(-x),
                        other => return Err(format!("line {}: cannot negate {}", line, other.type_name())),
                    });
                }
                Op::Not => {
                    let v = pop(&mut stack);
                    match v {
                        Value::Bool(b) => stack.push(Value::Bool(!b)),
                        other => return Err(format!("line {}: 'not' takes a bool ({} found)", line, other.type_name())),
                    }
                }
                Op::Add => {
                    let r = pop(&mut stack);
                    let l = pop(&mut stack);
                    stack.push(runtime::add(l, r, span)?);
                }
                Op::Sub | Op::Mul | Op::Div => {
                    let r = pop(&mut stack);
                    let l = pop(&mut stack);
                    let op = match &chunk.code[ip] {
                        Op::Sub => BinOp::Sub,
                        Op::Mul => BinOp::Mul,
                        _ => BinOp::Div,
                    };
                    stack.push(runtime::arith(&op, l, r, span, &mut self.channel)?);
                }
                Op::Eq => {
                    let r = pop(&mut stack);
                    let l = pop(&mut stack);
                    stack.push(Value::Bool(runtime::equals(&l, &r, span)?));
                }
                Op::Ne => {
                    let r = pop(&mut stack);
                    let l = pop(&mut stack);
                    stack.push(Value::Bool(!runtime::equals(&l, &r, span)?));
                }
                Op::Lt | Op::Gt | Op::Le | Op::Ge => {
                    let r = pop(&mut stack);
                    let l = pop(&mut stack);
                    let op = match &chunk.code[ip] {
                        Op::Lt => BinOp::Lt,
                        Op::Gt => BinOp::Gt,
                        Op::Le => BinOp::Le,
                        _ => BinOp::Ge,
                    };
                    stack.push(runtime::order(&op, l, r, span)?);
                }
                Op::LoadVar(i) => {
                    let v = self.load(&chunk.names[*i], scopes, line)?;
                    stack.push(v);
                }
                Op::DefineVar(i) => {
                    let v = pop(&mut stack);
                    self.define(&chunk.names[*i], v, scopes);
                }
                Op::AssignVar(i) => {
                    let v = pop(&mut stack);
                    self.assign(&chunk.names[*i], v, scopes);
                }
                Op::EnterScope => scopes.push(HashMap::new()),
                Op::ExitScope => {
                    scopes.pop();
                }
                Op::Jump(t) => {
                    ip = *t;
                    continue;
                }
                Op::JumpIfFalse(t) => {
                    let v = pop(&mut stack);
                    match v {
                        Value::Bool(false) => {
                            ip = *t;
                            continue;
                        }
                        Value::Bool(true) => {}
                        other => return Err(format!("line {}: condition must be bool ({} found)", line, other.type_name())),
                    }
                }
                Op::JumpIfFalsePeek(t) => {
                    match stack.last() {
                        Some(Value::Bool(false)) => {
                            ip = *t;
                            continue;
                        }
                        Some(Value::Bool(true)) => {}
                        other => {
                            let tn = other.map(|v| v.type_name()).unwrap_or("()");
                            return Err(format!("line {}: 'and' operand must be bool ({} found)", line, tn));
                        }
                    }
                }
                Op::JumpIfTruePeek(t) => {
                    match stack.last() {
                        Some(Value::Bool(true)) => {
                            ip = *t;
                            continue;
                        }
                        Some(Value::Bool(false)) => {}
                        other => {
                            let tn = other.map(|v| v.type_name()).unwrap_or("()");
                            return Err(format!("line {}: 'or' operand must be bool ({} found)", line, tn));
                        }
                    }
                }
                Op::BoolCheck => match stack.last() {
                    Some(Value::Bool(_)) => {}
                    other => {
                        let tn = other.map(|v| v.type_name()).unwrap_or("()");
                        return Err(format!("line {}: logical operand must be bool ({} found)", line, tn));
                    }
                },
                Op::Call(i, argc) => {
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack));
                    }
                    args.reverse();
                    let v = self.call(&chunk.names[*i], args, span)?;
                    stack.push(v);
                }
                Op::Return => {
                    return Ok(Some(stack.pop().unwrap_or(Value::Unit)));
                }
                Op::Try => {
                    // `?` — if the value is an error, return it from the current function (propagate).
                    let v = pop(&mut stack);
                    if matches!(v, Value::Err(_)) {
                        return Ok(Some(v));
                    }
                    stack.push(v);
                }
                Op::Print(argc) => {
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack));
                    }
                    args.reverse();
                    let parts: Vec<String> = args.iter().map(|v| v.to_string()).collect();
                    println!("{}", parts.join(" "));
                    stack.push(Value::Unit);
                }
                Op::MapNew => stack.push(Value::empty_map()),
                Op::Range => {
                    let hi = pop(&mut stack);
                    let lo = pop(&mut stack);
                    match (lo, hi) {
                        (Value::Int(a), Value::Int(b)) => stack.push(Value::Range(a, b)),
                        (a, b) => return Err(format!("line {}: range endpoints must be ints ({}..{})", line, a.type_name(), b.type_name())),
                    }
                }
                Op::Array(n) => {
                    let mut items = Vec::with_capacity(*n);
                    for _ in 0..*n {
                        items.push(pop(&mut stack));
                    }
                    items.reverse();
                    stack.push(Value::array(items));
                }
                Op::Index => {
                    let idx = pop(&mut stack);
                    let recv = pop(&mut stack);
                    stack.push(runtime::index_read(&recv, &idx, span, &mut self.channel)?);
                }
                Op::SetIndex => {
                    let val = pop(&mut stack);
                    let idx = pop(&mut stack);
                    let recv = pop(&mut stack);
                    if let Value::Array(a) = &recv {
                        runtime::borrow_check_mut(&self.borrows, a, "index assignment", span, &mut self.channel)?;
                    }
                    runtime::index_write(&recv, &idx, val, line)?;
                }
                Op::AddrOf(i) => {
                    let idx = pop(&mut stack);
                    let recv = pop(&mut stack);
                    let arr = match recv {
                        Value::Array(a) => a,
                        other => return Err(format!("line {}: can only take & of an array element yet ({})", line, other.type_name())),
                    };
                    let idx = match idx {
                        Value::Int(n) if n >= 0 => n as usize,
                        other => return Err(format!("line {}: pointer index must be a non-negative int ({})", line, other.type_name())),
                    };
                    let name = chunk.names[*i].clone();
                    stack.push(runtime::make_ptr(arr, idx, name, span, &mut self.channel)?);
                }
                Op::Deref => {
                    let v = pop(&mut stack);
                    match v {
                        Value::Ptr(p) => stack.push(runtime::deref_read(&p, span)?),
                        other => return Err(format!("line {}: cannot dereference {} (not a pointer)", line, other.type_name())),
                    }
                }
                Op::DerefSet => {
                    let val = pop(&mut stack);
                    let ptr = pop(&mut stack);
                    match ptr {
                        Value::Ptr(p) => runtime::deref_write(&p, val, span)?,
                        other => return Err(format!("line {}: cannot dereference {} (not a pointer)", line, other.type_name())),
                    }
                }
                Op::RawOp(i, argc) => {
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack));
                    }
                    args.reverse();
                    let name = &chunk.names[*i];
                    stack.push(runtime::raw_op(name, &args, span, &mut self.channel)?);
                }
                Op::BorrowShared(i) => {
                    let v = pop(&mut stack);
                    if let Value::Array(a) = &v {
                        let name = chunk.names[*i].clone();
                        runtime::borrow_register(&mut self.borrows, a, &name, "for iteration", span, &mut self.channel)?;
                    }
                }
                Op::BorrowRelease => {
                    let v = pop(&mut stack);
                    if let Value::Array(a) = &v {
                        runtime::borrow_release(&mut self.borrows, runtime::buf_id(a));
                    }
                }
                Op::ShowProvenance => {
                    let v = pop(&mut stack);
                    let record = runtime::provenance_record(&self.borrows, &v);
                    self.channel.info(span, record);
                }
                Op::Len => {
                    let v = pop(&mut stack);
                    let n = runtime::length(&v).ok_or_else(|| format!("line {}: no length for {} in the VM yet", line, v.type_name()))?;
                    stack.push(Value::Int(n));
                }
                Op::Field(i) => {
                    let recv = pop(&mut stack);
                    let name = &chunk.names[*i];
                    match &recv {
                        Value::Struct { fields, name: sname } => {
                            let v = fields
                                .borrow()
                                .get(name)
                                .cloned()
                                .ok_or_else(|| format!("line {}: {} has no field '{}'", line, sname, name))?;
                            stack.push(v);
                        }
                        _ if name == "len" => {
                            let n = runtime::length(&recv).ok_or_else(|| format!("line {}: no length for {}", line, recv.type_name()))?;
                            stack.push(Value::Int(n));
                        }
                        other => return Err(format!("line {}: {} has no field '{}'", line, other.type_name(), name)),
                    }
                }
                Op::Method(i, argc) => {
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack));
                    }
                    args.reverse();
                    let recv = pop(&mut stack);
                    let name = &chunk.names[*i];
                    if let Value::Array(a) = &recv {
                        if runtime::MUT_ARRAY_METHODS.contains(&name.as_str()) {
                            runtime::borrow_check_mut(&self.borrows, a, name, span, &mut self.channel)?;
                        }
                    }
                    let v = self.method(&recv, name, args, line)?;
                    stack.push(v);
                }
                Op::StructLit(ni, fnames) => {
                    let sname = chunk.names[*ni].clone();
                    let decl = self
                        .structs
                        .get(&sname)
                        .cloned()
                        .ok_or_else(|| format!("line {}: undefined struct '{}'", line, sname))?;
                    let mut vals = Vec::with_capacity(fnames.len());
                    for _ in 0..fnames.len() {
                        vals.push(pop(&mut stack));
                    }
                    vals.reverse();
                    let mut map = BTreeMap::new();
                    for (fi, v) in fnames.iter().zip(vals) {
                        let fname = &chunk.names[*fi];
                        if !decl.contains(fname) {
                            return Err(format!("line {}: {} has no field '{}'", line, sname, fname));
                        }
                        map.insert(fname.clone(), v);
                    }
                    stack.push(Value::Struct { name: sname, fields: Rc::new(RefCell::new(map)) });
                }
                Op::SetField(fi) => {
                    let val = pop(&mut stack);
                    let recv = pop(&mut stack);
                    let fname = &chunk.names[*fi];
                    match &recv {
                        Value::Struct { fields, name } => {
                            if !fields.borrow().contains_key(fname) {
                                return Err(format!("line {}: {} has no field '{}'", line, name, fname));
                            }
                            fields.borrow_mut().insert(fname.clone(), val);
                        }
                        other => return Err(format!("line {}: cannot assign field on {}", line, other.type_name())),
                    }
                }
                Op::EnumLit(ei, vi, argc) => {
                    let ename = chunk.names[*ei].clone();
                    let variant = chunk.names[*vi].clone();
                    match self.enums.get(&ename) {
                        None => return Err(format!("line {}: undefined enum '{}'", line, ename)),
                        Some(vs) => match vs.get(&variant) {
                            None => return Err(format!("line {}: {} has no variant '{}'", line, ename, variant)),
                            Some(&arity) if arity != *argc => {
                                return Err(format!("line {}: {}::{} expects {} arg(s), got {}", line, ename, variant, arity, argc))
                            }
                            _ => {}
                        },
                    }
                    let mut payload = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        payload.push(pop(&mut stack));
                    }
                    payload.reverse();
                    stack.push(Value::Enum { name: ename, variant, payload });
                }
                Op::TryMatch(pi, fail) => {
                    let subject = pop(&mut stack);
                    match match_pattern(&chunk.patterns[*pi], &subject) {
                        Some(binds) => {
                            for (n, v) in binds {
                                self.define(&n, v, scopes);
                            }
                        }
                        None => {
                            ip = *fail;
                            continue;
                        }
                    }
                }
                Op::MatchFail => return Err(format!("line {}: no match arm matched", line)),
                Op::Cout(argc) => {
                    // C++-style stream output: no auto-space, no trailing newline.
                    use std::io::Write;
                    let mut args = Vec::with_capacity(*argc);
                    for _ in 0..*argc {
                        args.push(pop(&mut stack));
                    }
                    args.reverse();
                    let s: String = args.iter().map(|v| v.to_string()).collect();
                    print!("{}", s);
                    let _ = std::io::stdout().flush();
                }
                Op::ReadInput => {
                    let tok = self
                        .next_input_token()
                        .ok_or_else(|| format!("line {}: cin — no more input", line))?;
                    self.channel.info(span, format!("stdin read: \"{}\"", tok));
                    stack.push(parse_input_token(&tok));
                }
            }
            ip += 1;
        }
        Ok(None)
    }

    fn call(&mut self, name: &str, args: Vec<Value>, span: crate::span::Span) -> Result<Value, String> {
        let line = span.line;
        if let Some(func) = self.funcs.get(name).cloned() {
            if args.len() != func.params.len() {
                return Err(format!("line {}: '{}' expects {} arg(s), got {}", line, name, func.params.len(), args.len()));
            }
            let mut frame = HashMap::new();
            for (p, a) in func.params.iter().zip(args) {
                frame.insert(p.clone(), a);
            }
            let mut scopes = vec![frame];
            return Ok(self.run_chunk(&func.chunk, &mut scopes)?.unwrap_or(Value::Unit));
        }
        // Not a user function — try a shared core builtin (len/str/int/abs/.../err/heap/set/strbuf).
        if let Some(res) = runtime::value_builtin(name, &args, span, &mut self.channel) {
            return res;
        }
        Err(format!("line {}: undefined function '{}' (or not yet supported by the VM)", line, name))
    }

}

impl Vm {
    /// Method dispatch for the VM's supported receivers (arrays, strings). Mirrors the tree-walker.
    fn method(&mut self, recv: &Value, name: &str, argv: Vec<Value>, line: usize) -> Result<Value, String> {
        match recv {
            Value::Array(a) => {
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
                            self.channel.warn(crate::span::Span::new(line, 1), "pop from empty array");
                            format!("line {}: pop from empty array", line)
                        })
                    }
                    "pop_front" => {
                        check_n(&argv, 0, name, line)?;
                        let mut b = a.borrow_mut();
                        if b.is_empty() {
                            return Err(format!("line {}: pop_front from empty array", line));
                        }
                        Ok(b.remove(0))
                    }
                    "contains" => {
                        check_n(&argv, 1, name, line)?;
                        Ok(Value::Bool(a.borrow().iter().any(|e| e == &argv[0])))
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
                        let mut acc = Value::Int(0);
                        for v in a.borrow().iter() {
                            acc = match (num(&acc), num(v)) {
                                (Some(_), Some(_)) => {
                                    if matches!(acc, Value::Int(_)) && matches!(v, Value::Int(_)) {
                                        Value::Int(l_int(&acc) + l_int(v))
                                    } else {
                                        Value::Float(num(&acc).unwrap() + num(v).unwrap())
                                    }
                                }
                                _ => return Err(format!("line {}: sum() takes a number array only ({})", line, v.type_name())),
                            };
                        }
                        Ok(acc)
                    }
                    "join" => {
                        check_n(&argv, 1, name, line)?;
                        let sep = match &argv[0] {
                            Value::Str(s) => s.clone(),
                            other => return Err(format!("line {}: join(sep) — sep must be str ({})", line, other.type_name())),
                        };
                        let parts: Vec<String> = a.borrow().iter().map(|v| v.to_string()).collect();
                        Ok(Value::Str(parts.join(&sep)))
                    }
                    _ => Err(format!("line {}: array method '{}' is not yet supported by the VM", line, name)),
                }
            }
            Value::Map(m) => match name {
                "get" => {
                    if argv.is_empty() || argv.len() > 2 {
                        return Err(format!("line {}: get(key[, default]) expects 1-2 args, got {}", line, argv.len()));
                    }
                    let key = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a map key", line, argv[0].type_name()))?;
                    match m.borrow().get(&key) {
                        Some(v) => Ok(v.clone()),
                        None => Ok(argv.get(1).cloned().unwrap_or(Value::Unit)),
                    }
                }
                "contains" => {
                    check_n(&argv, 1, name, line)?;
                    let key = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a map key", line, argv[0].type_name()))?;
                    Ok(Value::Bool(m.borrow().contains_key(&key)))
                }
                "remove" => {
                    check_n(&argv, 1, name, line)?;
                    let key = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a map key", line, argv[0].type_name()))?;
                    Ok(m.borrow_mut().remove(&key).unwrap_or(Value::Unit))
                }
                "keys" => {
                    check_n(&argv, 0, name, line)?;
                    Ok(Value::array(m.borrow().keys().map(MapKey::to_value).collect()))
                }
                "values" => {
                    check_n(&argv, 0, name, line)?;
                    Ok(Value::array(m.borrow().values().cloned().collect()))
                }
                _ => Err(format!("line {}: map method '{}' is not yet supported by the VM", line, name)),
            },
            Value::Str(s) => match name {
                "upper" => Ok(Value::Str(s.to_uppercase())),
                "lower" => Ok(Value::Str(s.to_lowercase())),
                "trim" => Ok(Value::Str(s.trim().to_string())),
                "chars" => Ok(Value::array(s.chars().map(|c| Value::Str(c.to_string())).collect())),
                "contains" => {
                    check_n(&argv, 1, name, line)?;
                    let sub = as_str(&argv[0], line)?;
                    Ok(Value::Bool(s.contains(&sub)))
                }
                "split" => {
                    check_n(&argv, 1, name, line)?;
                    let sep = as_str(&argv[0], line)?;
                    let parts: Vec<Value> = if sep.is_empty() {
                        s.chars().map(|c| Value::Str(c.to_string())).collect()
                    } else {
                        s.split(&sep as &str).map(|p| Value::Str(p.to_string())).collect()
                    };
                    Ok(Value::array(parts))
                }
                _ => Err(format!("line {}: string method '{}' is not yet supported by the VM", line, name)),
            },
            Value::StrBuf(b) => match name {
                "push" => {
                    check_n(&argv, 1, name, line)?;
                    match &argv[0] {
                        Value::Str(s) => b.borrow_mut().push_str(s),
                        Value::StrBuf(o) => {
                            let s = o.borrow().clone();
                            b.borrow_mut().push_str(&s);
                        }
                        other => return Err(format!("line {}: push(x) — x must be str/strbuf ({})", line, other.type_name())),
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
                _ => Err(format!("line {}: strbuf method '{}' is not yet supported by the VM", line, name)),
            },
            Value::Heap(h) => match name {
                "push" => {
                    check_n(&argv, 1, name, line)?;
                    crate::eval::heap_push(&mut h.borrow_mut(), argv[0].clone())?;
                    Ok(Value::Unit)
                }
                "pop" => {
                    check_n(&argv, 0, name, line)?;
                    crate::eval::heap_pop(&mut h.borrow_mut())?
                        .ok_or_else(|| format!("line {}: pop from empty heap", line))
                }
                "peek" => {
                    check_n(&argv, 0, name, line)?;
                    h.borrow().first().cloned().ok_or_else(|| format!("line {}: peek on empty heap", line))
                }
                _ => Err(format!("line {}: heap method '{}' is not yet supported by the VM", line, name)),
            },
            Value::Set(s) => match name {
                "add" => {
                    check_n(&argv, 1, name, line)?;
                    let k = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a set element", line, argv[0].type_name()))?;
                    s.borrow_mut().insert(k);
                    Ok(Value::Unit)
                }
                "contains" => {
                    check_n(&argv, 1, name, line)?;
                    let k = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a set element", line, argv[0].type_name()))?;
                    Ok(Value::Bool(s.borrow().contains(&k)))
                }
                "remove" => {
                    check_n(&argv, 1, name, line)?;
                    let k = argv[0].as_key().ok_or_else(|| format!("line {}: {} cannot be a set element", line, argv[0].type_name()))?;
                    Ok(Value::Bool(s.borrow_mut().remove(&k)))
                }
                "items" => {
                    check_n(&argv, 0, name, line)?;
                    Ok(Value::array(s.borrow().iter().map(MapKey::to_value).collect()))
                }
                _ => Err(format!("line {}: set method '{}' is not yet supported by the VM", line, name)),
            },
            Value::Struct { name: sname, .. } => {
                let sname = sname.clone();
                self.struct_method(recv.clone(), &sname, name, argv, line)
            }
            other => Err(format!("line {}: {} has no method '{}' in the VM", line, other.type_name(), name)),
        }
    }

    /// User-defined struct method: bind the receiver to the first parameter `self` (reference
    /// semantics), remaining params follow argv. Mirrors the tree-walker (v0.16 impl / v0.23 VM).
    fn struct_method(&mut self, recv: Value, sname: &str, mname: &str, argv: Vec<Value>, line: usize) -> Result<Value, String> {
        let func = self
            .methods
            .get(sname)
            .and_then(|t| t.get(mname))
            .cloned()
            .ok_or_else(|| format!("line {}: {} has no method '{}'", line, sname, mname))?;
        if func.params.is_empty() {
            return Err(format!("line {}: method '{}.{}' has no self parameter", line, sname, mname));
        }
        let want = func.params.len() - 1;
        if argv.len() != want {
            return Err(format!("line {}: method '{}.{}' expects {} arg(s), got {}", line, sname, mname, want, argv.len()));
        }
        let mut frame = HashMap::new();
        frame.insert(func.params[0].clone(), recv);
        for (p, v) in func.params[1..].iter().zip(argv) {
            frame.insert(p.clone(), v);
        }
        let mut scopes = vec![frame];
        Ok(self.run_chunk(&func.chunk, &mut scopes)?.unwrap_or(Value::Unit))
    }
}

fn as_index(v: &Value, len: usize, line: usize) -> Result<usize, String> {
    match v {
        Value::Int(n) if *n >= 0 && (*n as usize) < len => Ok(*n as usize),
        Value::Int(n) => Err(format!("line {}: index {} out of range (len {})", line, n, len)),
        other => Err(format!("line {}: index must be int ({} found)", line, other.type_name())),
    }
}

fn as_str(v: &Value, line: usize) -> Result<String, String> {
    match v {
        Value::Str(s) => Ok(s.clone()),
        other => Err(format!("line {}: expected str ({} found)", line, other.type_name())),
    }
}

fn check_n(argv: &[Value], n: usize, name: &str, line: usize) -> Result<(), String> {
    if argv.len() != n {
        Err(format!("line {}: {}() expects {} arg(s), got {}", line, name, n, argv.len()))
    } else {
        Ok(())
    }
}

fn pop(stack: &mut Vec<Value>) -> Value {
    stack.pop().unwrap_or(Value::Unit)
}

fn num(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

fn l_int(v: &Value) -> i64 {
    match v {
        Value::Int(n) => *n,
        _ => 0,
    }
}


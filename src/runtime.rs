//! Shared runtime: value-level operations used by BOTH backends — the tree-walker (eval.rs) and the
//! bytecode VM (vm.rs). One source of truth for operator semantics and core builtins, so the two
//! backends can never diverge (this also removed the earlier arithmetic/eq duplication).

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

use crate::ast::BinOp;
use crate::lumen::Channel;
use crate::span::Span;
use crate::value::{value_cmp, ArrayRef, PtrData, Value};

// ---- operators ----

pub fn add(l: Value, r: Value, span: Span) -> Result<Value, String> {
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

pub fn arith(op: &BinOp, l: Value, r: Value, span: Span, ch: &mut Channel) -> Result<Value, String> {
    if let (Value::Int(a), Value::Int(b)) = (&l, &r) {
        let (a, b) = (*a, *b);
        return Ok(match op {
            BinOp::Sub => Value::Int(a - b),
            BinOp::Mul => Value::Int(a * b),
            BinOp::Div => {
                if b == 0 {
                    ch.warn(span, "division by zero — runtime error");
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
                    ch.warn(span, "division by zero (0.0) — runtime error");
                    return Err(format!("line {}: division by zero", span.line));
                }
                Value::Float(x / y)
            }
            _ => unreachable!(),
        }),
        _ => Err(format!("line {}: cannot do arithmetic on {} and {}", span.line, l.type_name(), r.type_name())),
    }
}

pub fn equals(l: &Value, r: &Value, span: Span) -> Result<bool, String> {
    if let (Some(x), Some(y)) = (to_f(l), to_f(r)) {
        return Ok(x == y);
    }
    if std::mem::discriminant(l) == std::mem::discriminant(r) {
        return Ok(l == r);
    }
    Err(format!("line {}: {} and {} are not comparable", span.line, l.type_name(), r.type_name()))
}

pub fn order(op: &BinOp, l: Value, r: Value, span: Span) -> Result<Value, String> {
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

// ---- core value builtins (shared) ----

/// Core builtins that operate purely on already-evaluated values. Returns Some(result) if `name` is a
/// core builtin, None otherwise (→ user function). `print` and the tensor builtins are backend-specific
/// and handled by the caller. (std/heap/set/strbuf are created here; std gating is the caller's job.)
/// Normalize an index — a negative index counts from the end (v0.44, `xs[-1]` = last). Out of range → Err.
pub fn norm_index(n: i64, len: usize, line: usize) -> Result<usize, String> {
    let i = if n < 0 { n + len as i64 } else { n };
    if i < 0 || i as usize >= len {
        Err(format!("line {}: index {} out of range (len {})", line, n, len))
    } else {
        Ok(i as usize)
    }
}

/// Normalize a slice range `[a..b)` — endpoints may be negative (from the end); requires 0 ≤ a ≤ b ≤ len.
/// Strict bounds (no silent clamping): the checked path *blocks*, consistent with indexing (§3.2).
pub fn norm_range(a: i64, b: i64, len: usize, line: usize) -> Result<(usize, usize), String> {
    let na = if a < 0 { a + len as i64 } else { a };
    let nb = if b < 0 { b + len as i64 } else { b };
    if na < 0 || nb < na || nb as usize > len {
        Err(format!("line {}: slice {}..{} out of range (len {})", line, a, b, len))
    } else {
        Ok((na as usize, nb as usize))
    }
}

/// Shared index read — arrays/strings/maps, int or range (slice) index; negative indexes count from the
/// end (v0.44). Both backends call this (single semantics source). Slices copy — the copy is illuminated.
pub fn index_read(recv: &Value, idx: &Value, span: Span, ch: &mut Channel) -> Result<Value, String> {
    let line = span.line;
    match (recv, idx) {
        (Value::Array(a), Value::Int(n)) => {
            let b = a.borrow();
            match norm_index(*n, b.len(), line) {
                Ok(i) => Ok(b[i].clone()),
                Err(e) => {
                    ch.warn(span, format!("index {} out of range (len {})", n, b.len()));
                    Err(e)
                }
            }
        }
        (Value::Array(a), Value::Range(x, y)) => {
            let b = a.borrow();
            let (i, j) = norm_range(*x, *y, b.len(), line)?;
            ch.info(span, format!("slice [{}..{}) → new heap vector · {} elems (copy)", i, j, j - i));
            Ok(Value::array(b[i..j].to_vec()))
        }
        (Value::Str(s), Value::Int(n)) => {
            let chars: Vec<char> = s.chars().collect();
            match norm_index(*n, chars.len(), line) {
                Ok(i) => Ok(Value::Str(chars[i].to_string())),
                Err(e) => {
                    ch.warn(span, format!("index {} out of range (len {})", n, chars.len()));
                    Err(e)
                }
            }
        }
        (Value::Str(s), Value::Range(x, y)) => {
            let chars: Vec<char> = s.chars().collect();
            let (i, j) = norm_range(*x, *y, chars.len(), line)?;
            ch.info(span, format!("slice [{}..{}) → str · {} chars (copy)", i, j, j - i));
            Ok(Value::Str(chars[i..j].iter().collect()))
        }
        (Value::Map(m), _) => {
            let key = idx.as_key().ok_or_else(|| format!("line {}: {} cannot be used as a map key", line, idx.type_name()))?;
            m.borrow()
                .get(&key)
                .cloned()
                .ok_or_else(|| format!("line {}: key not found: {}", line, idx))
        }
        (Value::Array(_), other) | (Value::Str(_), other) => {
            Err(format!("line {}: index must be int or range (found {})", line, other.type_name()))
        }
        (other, _) => Err(format!("line {}: {} is not indexable", line, other.type_name())),
    }
}

/// Shared index write — negative indexes normalize like reads; slices are not assignable.
pub fn index_write(recv: &Value, idx: &Value, val: Value, line: usize) -> Result<(), String> {
    match (recv, idx) {
        (Value::Array(a), Value::Int(n)) => {
            let len = a.borrow().len();
            let i = norm_index(*n, len, line)?;
            a.borrow_mut()[i] = val;
            Ok(())
        }
        (Value::Array(_), Value::Range(..)) | (Value::Str(_), Value::Range(..)) => {
            Err(format!("line {}: cannot assign to a slice", line))
        }
        (Value::Map(m), _) => {
            let key = idx.as_key().ok_or_else(|| format!("line {}: {} cannot be used as a map key", line, idx.type_name()))?;
            m.borrow_mut().insert(key, val);
            Ok(())
        }
        (other, _) => Err(format!("line {}: {} does not support index assignment", line, other.type_name())),
    }
}

/// The single string-path argument of an fs builtin (std/fs, v0.43).
fn fs_path(argv: &[Value], name: &str, line: usize) -> Result<String, String> {
    match argv {
        [Value::Str(p)] => Ok(p.clone()),
        _ => Err(format!("line {}: {}(path) takes one string path", line, name)),
    }
}

/// An I/O failure as an error-value (so scripts can `?`-propagate or is_err-check it).
fn fs_err(name: &str, path: &str, e: &std::io::Error) -> Value {
    Value::Err(Box::new(Value::Str(format!("{} \"{}\": {}", name, path, e))))
}

pub fn value_builtin(name: &str, argv: &[Value], span: Span, ch: &mut Channel) -> Option<Result<Value, String>> {
    let line = span.line;
    let one = |argv: &[Value]| -> Result<Value, String> {
        if argv.len() == 1 {
            Ok(argv[0].clone())
        } else {
            Err(format!("line {}: {}() expects 1 arg, got {}", line, name, argv.len()))
        }
    };
    let result: Result<Value, String> = match name {
        "heap" => {
            ch.info(span, "heap (priority queue, mutable)");
            Ok(Value::empty_heap())
        }
        "set" => {
            ch.info(span, "heap set (mutable)");
            Ok(Value::Set(Rc::new(RefCell::new(BTreeSet::new()))))
        }
        "strbuf" => {
            ch.info(span, "string builder (amortized O(1) append — avoids `s = s + c` O(n²))");
            Ok(Value::StrBuf(Rc::new(RefCell::new(String::new()))))
        }
        // ---- file I/O (std/fs, v0.43) — I/O is a cost: every read/write illuminates its byte count.
        // I/O failures are error-*values* (Zig-style union, works with `?`/is_err), not hard errors.
        "read_file" => fs_path(argv, name, line).map(|p| match std::fs::read_to_string(&p) {
            Ok(s) => {
                ch.info(span, format!("fs read \"{}\" · {} B", p, s.len()));
                Value::Str(s)
            }
            Err(e) => fs_err(name, &p, &e),
        }),
        "read_lines" => fs_path(argv, name, line).map(|p| match std::fs::read_to_string(&p) {
            Ok(s) => {
                ch.info(span, format!("fs read \"{}\" · {} B → lines", p, s.len()));
                Value::array(s.lines().map(|l| Value::Str(l.to_string())).collect())
            }
            Err(e) => fs_err(name, &p, &e),
        }),
        "write_file" | "append_file" => {
            if argv.len() != 2 {
                Err(format!("line {}: {}(path, content) expects 2 args, got {}", line, name, argv.len()))
            } else {
                fs_path(&argv[..1], name, line).map(|p| {
                    let content = argv[1].to_string();
                    let res = if name == "write_file" {
                        std::fs::write(&p, &content)
                    } else {
                        use std::io::Write;
                        std::fs::OpenOptions::new().create(true).append(true).open(&p).and_then(|mut f| f.write_all(content.as_bytes()))
                    };
                    match res {
                        Ok(()) => {
                            let verb = if name == "write_file" { "write" } else { "append" };
                            ch.info(span, format!("fs {} \"{}\" · {} B", verb, p, content.len()));
                            Value::Unit
                        }
                        Err(e) => fs_err(name, &p, &e),
                    }
                })
            }
        }
        "remove_file" => fs_path(argv, name, line).map(|p| match std::fs::remove_file(&p) {
            Ok(()) => {
                ch.info(span, format!("fs remove \"{}\"", p));
                Value::Unit
            }
            Err(e) => fs_err(name, &p, &e),
        }),
        "file_exists" => fs_path(argv, name, line).map(|p| Value::Bool(std::path::Path::new(&p).exists())),
        "len" => one(argv).and_then(|v| {
            length(&v)
                .map(Value::Int)
                .ok_or_else(|| format!("line {}: len() cannot be applied to {}", line, v.type_name()))
        }),
        "int" => to_int(argv, line),
        "float" => one(argv).and_then(|v| match v {
            Value::Int(n) => Ok(Value::Float(n as f64)),
            Value::Float(x) => Ok(Value::Float(x)),
            Value::Str(s) => s.trim().parse().map(Value::Float).map_err(|_| format!("line {}: failed to parse float '{}'", line, s)),
            other => Err(format!("line {}: float() cannot be applied to {}", line, other.type_name())),
        }),
        "str" => one(argv).map(|v| Value::Str(v.to_string())),
        "abs" => one(argv).and_then(|v| match v {
            Value::Int(n) => Ok(Value::Int(n.abs())),
            Value::Float(x) => Ok(Value::Float(x.abs())),
            other => Err(format!("line {}: abs() takes numbers only (found {})", line, other.type_name())),
        }),
        "sqrt" => one_num(argv, name, line).map(|x| Value::Float(x.sqrt())),
        "floor" => one_num(argv, name, line).map(|x| Value::Int(x.floor() as i64)),
        "ceil" => one_num(argv, name, line).map(|x| Value::Int(x.ceil() as i64)),
        "pow" => {
            if argv.len() != 2 {
                Err(format!("line {}: pow() expects 2 args, got {}", line, argv.len()))
            } else {
                pow(&argv[0], &argv[1], line)
            }
        }
        "min" | "max" => {
            if argv.is_empty() {
                Err(format!("line {}: {}() needs at least 1 arg", line, name))
            } else {
                let want_less = name == "min";
                let mut best = argv[0].clone();
                let mut err = None;
                for v in &argv[1..] {
                    match value_cmp(v, &best) {
                        Ok(ord) => {
                            if (want_less && ord == Ordering::Less) || (!want_less && ord == Ordering::Greater) {
                                best = v.clone();
                            }
                        }
                        Err(e) => {
                            err = Some(format!("line {}: {}", line, e));
                            break;
                        }
                    }
                }
                match err {
                    Some(e) => Err(e),
                    None => Ok(best),
                }
            }
        }
        "hex" => one_int(argv, name, line).map(|n| Value::Str(radix_str(n, 16, "0x"))),
        "bin" => one_int(argv, name, line).map(|n| Value::Str(radix_str(n, 2, "0b"))),
        "ord" => one(argv).and_then(|v| match v {
            Value::Str(s) if s.chars().count() == 1 => Ok(Value::Int(s.chars().next().unwrap() as i64)),
            _ => Err(format!("line {}: ord() takes a single-character string only", line)),
        }),
        "chr" => one_int(argv, name, line).and_then(|n| {
            u32::try_from(n)
                .ok()
                .and_then(char::from_u32)
                .map(|c| Value::Str(c.to_string()))
                .ok_or_else(|| format!("line {}: chr({}) invalid code point", line, n))
        }),
        "err" => one(argv).map(|v| Value::Err(Box::new(v))),
        "is_err" => one(argv).map(|v| Value::Bool(matches!(v, Value::Err(_)))),
        "err_msg" => one(argv).map(|v| match v {
            Value::Err(p) => *p,
            other => other,
        }),
        "assert" => {
            if argv.is_empty() || argv.len() > 2 {
                Err(format!("line {}: assert(cond[, msg])", line))
            } else {
                match argv[0].truthy() {
                    None => Err(format!("line {}: assert condition must be bool", line)),
                    Some(true) => Ok(Value::Unit),
                    Some(false) => {
                        let msg = argv.get(1).map(|v| v.to_string()).unwrap_or_else(|| "assertion failed".into());
                        ch.warn(span, format!("assertion failed: {}", msg));
                        Err(format!("line {}: assertion failed: {}", line, msg))
                    }
                }
            }
        }
        _ => return None,
    };
    Some(result)
}

// ---- shared helpers (moved here so both backends use the same logic) ----

pub fn to_f(v: &Value) -> Option<f64> {
    match v {
        Value::Int(n) => Some(*n as f64),
        Value::Float(x) => Some(*x),
        _ => None,
    }
}

fn one_num(argv: &[Value], name: &str, line: usize) -> Result<f64, String> {
    if argv.len() != 1 {
        return Err(format!("line {}: {}() expects 1 arg, got {}", line, name, argv.len()));
    }
    to_f(&argv[0]).ok_or_else(|| format!("line {}: {}() takes numbers only (found {})", line, name, argv[0].type_name()))
}

fn one_int(argv: &[Value], name: &str, line: usize) -> Result<i64, String> {
    if argv.len() != 1 {
        return Err(format!("line {}: {}() expects 1 arg, got {}", line, name, argv.len()));
    }
    match &argv[0] {
        Value::Int(n) => Ok(*n),
        other => Err(format!("line {}: {}() takes an int (found {})", line, name, other.type_name())),
    }
}

pub fn length(v: &Value) -> Option<i64> {
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

pub fn to_int(argv: &[Value], line: usize) -> Result<Value, String> {
    match argv {
        [v] => match v {
            Value::Int(n) => Ok(Value::Int(*n)),
            Value::Float(x) => Ok(Value::Int(*x as i64)),
            Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
            Value::Str(s) => s.trim().parse().map(Value::Int).map_err(|_| format!("line {}: failed to parse int '{}'", line, s)),
            other => Err(format!("line {}: int() cannot be applied to {}", line, other.type_name())),
        },
        [Value::Str(s), Value::Int(base)] => {
            let b = u32::try_from(*base).map_err(|_| format!("line {}: invalid radix {}", line, base))?;
            let t = s.trim().trim_start_matches("0x").trim_start_matches("0b");
            i64::from_str_radix(t, b).map(Value::Int).map_err(|_| format!("line {}: int('{}', {}) failed to parse", line, s, base))
        }
        _ => Err(format!("line {}: int(x) or int(str, base)", line)),
    }
}

pub fn pow(b: &Value, e: &Value, line: usize) -> Result<Value, String> {
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

pub fn radix_str(n: i64, radix: u32, prefix: &str) -> String {
    let digits = |mut x: u64| -> String {
        if x == 0 {
            return "0".into();
        }
        let mut s = Vec::new();
        while x > 0 {
            let d = (x % radix as u64) as u32;
            s.push(std::char::from_digit(d, radix).unwrap());
            x /= radix as u64;
        }
        s.iter().rev().collect()
    };
    if n < 0 {
        format!("-{}{}", prefix, digits((-n) as u64))
    } else {
        format!("{}{}", prefix, digits(n as u64))
    }
}

// ---- memory model (pointers + provenance) — shared by both backends (§3.1) ----

/// Buffer identity for borrow/provenance tracking — the `Rc` address.
pub fn buf_id(a: &ArrayRef) -> usize {
    Rc::as_ptr(a) as *const () as usize
}

/// Build a pointer to `array[idx]` with provenance illumination (§3.1). Bounds-checked (blocks on overrun).
pub fn make_ptr(array: ArrayRef, idx: usize, name: String, span: Span, ch: &mut Channel) -> Result<Value, String> {
    let len = array.borrow().len();
    if idx >= len {
        ch.warn(span, format!("&{}[{}] out of bounds (len {})", name, idx, len));
        return Err(format!("line {}: &{}[{}] out of bounds (len {})", span.line, name, idx, len));
    }
    ch.info(span, format!("ptr → {}[{}] · origin {} · extent 0..{}", name, idx, name, len));
    Ok(Value::Ptr(Rc::new(PtrData { origin: array, index: idx, origin_name: name })))
}

/// Read through a pointer (checked — a dangling pointer, past the live extent, errors).
pub fn deref_read(p: &PtrData, span: Span) -> Result<Value, String> {
    let a = p.origin.borrow();
    a.get(p.index)
        .cloned()
        .ok_or_else(|| format!("line {}: dangling pointer — {}[{}] out of range (len {})", span.line, p.origin_name, p.index, a.len()))
}

/// Write through a pointer (checked).
pub fn deref_write(p: &PtrData, v: Value, span: Span) -> Result<(), String> {
    let mut a = p.origin.borrow_mut();
    if p.index >= a.len() {
        return Err(format!("line {}: dangling pointer write — {}[{}] out of range", span.line, p.origin_name, p.index));
    }
    a[p.index] = v;
    Ok(())
}

// ---- raw namespace (memory model §3.2/§3.5) — shared by both backends ----
// Low-level ops that *illuminate* bounds instead of blocking them (the memory side of D-novelty:
// "a check that blocks vs a check that illuminates"). Tracking stays on; an overrun is WARNed
// (responsibility shifts to the caller), not refused. Honesty (§3.7): no real addresses, so the
// silicon overrun is modeled (clamped to the live buffer); extent is in elements, not bytes.
// Takes already-evaluated arguments so the tree-walker and the VM call the identical logic.

fn raw_ptr(v: &Value, span: Span) -> Result<Rc<PtrData>, String> {
    match v {
        Value::Ptr(p) => Ok(p.clone()),
        other => Err(format!("line {}: raw op expects a pointer (found {})", span.line, other.type_name())),
    }
}

fn raw_count(v: &Value, span: Span) -> Result<usize, String> {
    match v {
        Value::Int(n) if *n >= 0 => Ok(*n as usize),
        other => Err(format!("line {}: raw count must be a non-negative int (found {})", span.line, other.type_name())),
    }
}

/// Illuminate an `n`-element raw access through `p`, given `avail` elements left in its extent.
fn raw_bounds(ch: &mut Channel, span: Span, op: &str, p: &PtrData, n: usize, avail: usize) {
    if n <= avail {
        ch.info(span, format!("raw.{} {} elem ⊂ {}.extent ({} avail) — bounds checked, safe", op, n, p.origin_name, avail));
    } else {
        ch.warn(span, format!("raw.{} {} elem > {}.extent ({} avail) — {} elem overrun possible (responsibility: caller)", op, n, p.origin_name, avail, n - avail));
    }
}

pub fn raw_op(name: &str, argv: &[Value], span: Span, ch: &mut Channel) -> Result<Value, String> {
    match name {
        "read" => {
            if argv.len() != 2 {
                return Err(format!("line {}: raw.read(p, n) takes 2 args, got {}", span.line, argv.len()));
            }
            let p = raw_ptr(&argv[0], span)?;
            let n = raw_count(&argv[1], span)?;
            let len = p.origin.borrow().len();
            let avail = len.saturating_sub(p.index);
            raw_bounds(ch, span, "read", &p, n, avail);
            let end = (p.index + n).min(len); // clamp — honest model of the overrun
            let out: Vec<Value> = p.origin.borrow()[p.index..end].to_vec();
            Ok(Value::Array(Rc::new(RefCell::new(out))))
        }
        "write" => {
            if argv.len() != 2 {
                return Err(format!("line {}: raw.write(p, vals) takes 2 args, got {}", span.line, argv.len()));
            }
            let p = raw_ptr(&argv[0], span)?;
            let src = match &argv[1] {
                Value::Array(a) => a.borrow().clone(),
                other => return Err(format!("line {}: raw.write expects an array of values (found {})", span.line, other.type_name())),
            };
            let n = src.len();
            let len = p.origin.borrow().len();
            let avail = len.saturating_sub(p.index);
            raw_bounds(ch, span, "write", &p, n, avail);
            let mut buf = p.origin.borrow_mut();
            for (k, v) in src.into_iter().enumerate() {
                let pos = p.index + k;
                if pos < buf.len() {
                    buf[pos] = v; // clamp — past-extent writes are illuminated, not performed
                }
            }
            Ok(Value::Int(n.min(avail) as i64)) // elements actually written
        }
        "memcpy" => {
            if argv.len() != 3 {
                return Err(format!("line {}: raw.memcpy(dst, src, n) takes 3 args, got {}", span.line, argv.len()));
            }
            let dst = raw_ptr(&argv[0], span)?;
            let src = raw_ptr(&argv[1], span)?;
            let n = raw_count(&argv[2], span)?;
            let savail = src.origin.borrow().len().saturating_sub(src.index);
            let davail = dst.origin.borrow().len().saturating_sub(dst.index);
            if n <= savail && n <= davail {
                ch.info(span, format!("raw.memcpy {} elem ⊂ src.extent ({} avail) ∩ dst.extent ({} avail) — bounds checked, safe", n, savail, davail));
            } else if savail <= davail {
                ch.warn(span, format!("raw.memcpy {} elem > src.extent ({} avail) — {} elem overrun possible (responsibility: caller)", n, savail, n - savail));
            } else {
                ch.warn(span, format!("raw.memcpy {} elem > dst.extent ({} avail) — {} elem overrun possible (responsibility: caller)", n, davail, n - davail));
            }
            let copy = n.min(savail).min(davail); // clamp to both live extents
            // Read the source chunk first (releases the src borrow) so dst==src buffers don't double-borrow.
            let chunk: Vec<Value> = src.origin.borrow()[src.index..src.index + copy].to_vec();
            let mut d = dst.origin.borrow_mut();
            for (k, v) in chunk.into_iter().enumerate() {
                d[dst.index + k] = v;
            }
            Ok(Value::Int(copy as i64)) // elements actually copied
        }
        other => Err(format!("line {}: unknown raw operation 'raw.{}' (have read/write/memcpy)", span.line, other)),
    }
}

// ---- borrow gradient + provenance record (memory model §3.3/§3.4) — shared by both backends ----
// The borrow table maps a buffer's identity (Rc address) → (shared-borrow count, origin name). A
// `for` loop holds one shared borrow for its body; a mutation of that buffer while shared-borrowed
// is a `&mut`-while-`&` conflict (iterator invalidation), illuminated and blocked. Lifetime is
// exactly the loop, so this is false-positive-free (principle 6).

/// Array methods that need *exclusive* access (mutate the buffer) — guarded by the borrow gradient.
pub const MUT_ARRAY_METHODS: &[&str] = &[
    "push", "push_front", "pop", "pop_front", "insert", "remove", "reverse", "sort", "clear",
];

/// The live borrow state of one buffer: shared count XOR one exclusive claim (§3.3 D2/D3).
#[derive(Clone, Default)]
pub struct BorrowState {
    pub shared: usize,
    pub exclusive: bool,
    pub name: String,
}

pub type Borrows = HashMap<usize, BorrowState>;

/// Register a shared borrow of `a` (a `for` iteration or an explicit `&x`, v0.48). Fails if the buffer
/// is exclusively borrowed — & XOR &mut. Illuminates the runtime-guard tier.
pub fn borrow_register(borrows: &mut Borrows, a: &ArrayRef, name: &str, what: &str, span: Span, ch: &mut Channel) -> Result<(), String> {
    let entry = borrows.entry(buf_id(a)).or_insert_with(|| BorrowState { name: name.to_string(), ..Default::default() });
    if entry.exclusive {
        let n = entry.name.clone();
        ch.warn(span, format!("borrow conflict: & of {} while {} is exclusively borrowed (&mut) — &mut XOR &", n, n));
        return Err(format!("line {}: borrow conflict — shared borrow of {} while it is exclusively borrowed", span.line, n));
    }
    entry.shared += 1;
    ch.info(span, format!("shared borrow of {} {} (runtime guard — XOR exclusive)", name, what));
    Ok(())
}

/// Register an *exclusive* borrow (`&mut x`, v0.48). Fails if any borrow is live on the buffer.
/// Honesty: this runtime tier guards borrow-vs-borrow conflicts and iterations; mutations while
/// exclusively borrowed are assumed to go through the borrow (path attribution is the static tier's job).
pub fn borrow_register_mut(borrows: &mut Borrows, a: &ArrayRef, name: &str, span: Span, ch: &mut Channel) -> Result<(), String> {
    let entry = borrows.entry(buf_id(a)).or_insert_with(|| BorrowState { name: name.to_string(), ..Default::default() });
    if entry.exclusive || entry.shared > 0 {
        let n = entry.name.clone();
        ch.warn(span, format!("borrow conflict: &mut of {} while it is already borrowed — &mut XOR &", n));
        return Err(format!("line {}: borrow conflict — &mut of {} while it is already borrowed", span.line, n));
    }
    entry.exclusive = true;
    ch.info(span, format!("exclusive borrow (&mut) of {} (runtime guard tier)", name));
    Ok(())
}

/// Release a previously registered shared borrow (called on every loop exit path / scope end).
pub fn borrow_release(borrows: &mut Borrows, id: usize) {
    if let Some(e) = borrows.get_mut(&id) {
        e.shared = e.shared.saturating_sub(1);
        if e.shared == 0 && !e.exclusive {
            borrows.remove(&id);
        }
    }
}

/// Release an exclusive borrow at scope end (v0.48).
pub fn borrow_release_mut(borrows: &mut Borrows, id: usize) {
    if let Some(e) = borrows.get_mut(&id) {
        e.exclusive = false;
        if e.shared == 0 {
            borrows.remove(&id);
        }
    }
}

/// Reject a mutation of `a` while it is shared-borrowed. Illuminates the conflict (WARN) then errors —
/// checked safety blocks (§3.2); `@trust` overrides (v0.48).
pub fn borrow_check_mut(borrows: &Borrows, a: &ArrayRef, op: &str, span: Span, ch: &mut Channel) -> Result<(), String> {
    if let Some(st) = borrows.get(&buf_id(a)) {
        if st.shared > 0 {
            let name = st.name.clone();
            ch.warn(span, format!("borrow conflict: {} on {} needs &mut, but {} is shared-borrowed — &mut XOR &", op, name, name));
            return Err(format!("line {}: borrow conflict — {} mutates {} while it is shared-borrowed (iterator invalidation)", span.line, op, name));
        }
    }
    Ok(())
}

/// The current access state of a buffer, derived from the live borrow table (§3.4 `access`).
pub fn access_of(borrows: &Borrows, id: usize) -> String {
    match borrows.get(&id) {
        Some(st) if st.exclusive => "exclusive (&mut)".to_string(),
        Some(st) if st.shared > 0 => format!("shared({})", st.shared),
        _ => "owner (no active borrow)".to_string(),
    }
}

/// Render the unified provenance record `{ origin, extent, alive, access }` for `@show provenance`
/// (§3.4, principle 5). Pointers carry a full record; arrays show their buffer; others have none.
pub fn provenance_record(borrows: &Borrows, v: &Value) -> String {
    match v {
        Value::Ptr(p) => {
            let len = p.origin.borrow().len();
            let alive = p.index < len;
            format!(
                "provenance: origin {} · extent 0..{} · pointee [{}] · alive {} · access {}",
                p.origin_name, len, p.index, alive, access_of(borrows, buf_id(&p.origin))
            )
        }
        Value::Array(a) => format!(
            "provenance: heap buffer · extent 0..{} · alive true · access {}",
            a.borrow().len(), access_of(borrows, buf_id(a))
        ),
        other => format!("provenance: {} is not a reference — provenance applies to pointers/buffers", other.type_name()),
    }
}

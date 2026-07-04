//! Bytecode for the wide VM (stage 2 of the roadmap — a real compile step).
//!
//! Stack-based. The tree-walking interpreter (eval.rs) remains the reference/fallback while this
//! backend grows toward parity. v0.19 covers the core subset (scalars, control flow, functions);
//! arrays/maps/struct/enum/match/tensor/IO come in later increments and currently compile to an error.

use crate::ast::Pattern;
use crate::span::Span;
use crate::value::Value;

/// A single stack-machine instruction. Indices point into the owning `Chunk`'s pools.
#[derive(Clone, Debug)]
pub enum Op {
    Const(usize), // push consts[i]
    True,
    False,
    Unit,
    Pop,

    Neg,
    Not,
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,

    LoadVar(usize),   // push value of names[i] (scope chain → globals)
    DefineVar(usize), // names[i] = pop() in the current scope (typed `let`)
    AssignVar(usize), // assign names[i] = pop() (search outward, else define)

    EnterScope,
    ExitScope,

    Jump(usize),            // ip = target
    JumpIfFalse(usize),     // v = pop(); requires bool; if !v ip = target
    JumpIfFalsePeek(usize), // requires bool; if top is false, ip = target (leaves it) — for `and`
    JumpIfTruePeek(usize),  // requires bool; if top is true, ip = target (leaves it) — for `or`
    BoolCheck,              // error if top is not bool (peek) — and/or operand typing

    Call(usize, usize), // call function names[i] with argc args
    Return,             // return top of stack
    Try,                // v = pop(); if v is Err, return it from the current function (`?` propagation)
    Print(usize),       // print argc values, space-joined + newline

    // data structures (v0.20)
    MapNew,               // push an empty map (v0.21)
    Range,                // pop hi, lo (ints) → push Value::Range (v0.44 — slices etc.)
    Array(usize),         // build an array from the top n stack values
    Index,                // pop idx, pop recv → push recv[idx] (array/str)
    SetIndex,             // pop val, pop idx, pop recv → recv[idx] = val (array)
    Len,                  // pop v → push length (array/str)
    Method(usize, usize), // pop argc args, pop recv → call method names[i] (array)
    Field(usize),         // pop recv → push field names[i] (.len or struct field)

    // types (v0.23)
    StructLit(usize, Vec<usize>), // struct names[i] { fields[..] } — pops one value per field name
    SetField(usize),              // pop val, pop recv (struct) → recv.field = val
    EnumLit(usize, usize, usize), // names[enum], names[variant], argc → build Enum
    TryMatch(usize, usize),       // pop subject; match patterns[i]; bind on success, else jump target
    MatchFail,                    // no arm matched in a match-expression → runtime error

    // I/O (v0.24)
    Cout(usize),                  // pop argc values → print concatenated (no auto-space/newline)
    ReadInput,                    // push the next whitespace token from stdin, auto-typed (int/float/str)

    // memory model (v0.29) — pointers
    AddrOf(usize), // pop idx, pop array → push a pointer to array[idx]; names[i] = origin name (provenance)
    Deref,         // pop ptr → push the pointee (checked read)
    DerefSet,      // pop val, pop ptr → write the pointee (checked)

    // memory model (v0.30) — raw namespace
    RawOp(usize, usize), // pop argc args → call raw operation names[i] (read/write/memcpy), bounds-illuminated

    // memory model (v0.31) — borrow gradient + provenance
    BorrowShared(usize), // pop array → register a shared borrow of its buffer for the loop; names[i] = origin name
    BorrowRelease,       // pop array → release one shared borrow of its buffer (loop exit)
    ShowProvenance,      // pop value → illuminate its provenance record (@show provenance)
}

/// A unit of compiled code: instructions + parallel spans + interned constants/names.
#[derive(Clone, Debug, Default)]
pub struct Chunk {
    pub code: Vec<Op>,
    pub spans: Vec<Span>,
    pub consts: Vec<Value>,
    pub names: Vec<String>,
    pub patterns: Vec<Pattern>,
}

impl Chunk {
    pub fn new() -> Self {
        Chunk::default()
    }

    /// Append an instruction with its source span; returns its index (for backpatching jumps).
    pub fn emit(&mut self, op: Op, span: Span) -> usize {
        self.code.push(op);
        self.spans.push(span);
        self.code.len() - 1
    }

    /// Intern a constant value, returning its index.
    pub fn add_const(&mut self, v: Value) -> usize {
        self.consts.push(v);
        self.consts.len() - 1
    }

    /// Intern a name (dedup), returning its index.
    pub fn add_name(&mut self, n: &str) -> usize {
        if let Some(i) = self.names.iter().position(|x| x == n) {
            return i;
        }
        self.names.push(n.to_string());
        self.names.len() - 1
    }

    /// Store a match pattern, returning its index.
    pub fn add_pattern(&mut self, p: Pattern) -> usize {
        self.patterns.push(p);
        self.patterns.len() - 1
    }
}

/// A compiled function: parameter names + its code chunk.
#[derive(Clone, Debug)]
pub struct CompiledFn {
    pub params: Vec<String>,
    pub chunk: Chunk,
}

/// A whole compiled program: the top-level `main` chunk + hoisted functions, struct/enum defs, and methods.
#[derive(Clone, Debug, Default)]
pub struct Program {
    pub main: Chunk,
    pub funcs: std::collections::HashMap<String, CompiledFn>,
    pub structs: std::collections::HashMap<String, Vec<String>>, // name → field names
    pub enums: std::collections::HashMap<String, std::collections::HashMap<String, usize>>, // name → (variant → arity)
    pub methods: std::collections::HashMap<String, std::collections::HashMap<String, CompiledFn>>, // struct → (method → fn)
}

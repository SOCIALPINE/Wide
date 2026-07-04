//! AST. Every node carries a Span — so the illumination points to the exact line.

use crate::span::Span;

#[derive(Clone, Debug)]
pub enum BinOp {
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
}

#[derive(Clone, Debug)]
pub enum Expr {
    Int(i64, Span),
    Float(f64, Span),
    Bool(bool, Span),
    Str(String, Span),
    Ident(String, Span),
    Array(Vec<Expr>, Span),
    Map(Span),
    Range(Box<Expr>, Box<Expr>, Span),
    Neg(Box<Expr>, Span),
    Not(Box<Expr>, Span),
    AddrOf(Box<Expr>, Span), // &lvalue — a pointer (&xs[i]) or a shared borrow claim (&xs) (§3.1/§3.3)
    AddrOfMut(Box<Expr>, Span), // &mut x — an exclusive borrow claim (v0.48, borrow gradient §3.3)
    Deref(Box<Expr>, Span),  // *ptr — read/write through a pointer
    And(Box<Expr>, Box<Expr>, Span),
    Or(Box<Expr>, Box<Expr>, Span),
    Binary(BinOp, Box<Expr>, Box<Expr>, Span),
    Call(String, Vec<Expr>, Span),
    Method(Box<Expr>, String, Vec<Expr>, Span),
    Field(Box<Expr>, String, Span),
    Index(Box<Expr>, Box<Expr>, Span),
    StructLit(String, Vec<(String, Expr)>, Span),
    EnumLit(String, String, Vec<Expr>, Span), // enum, variant, payload
    Match(Box<Expr>, Vec<(Pattern, Expr)>, Span), // match expression — arms are expressions
    Try(Box<Expr>, Span),                          // expr? — on error, propagate from the current function
    Lambda(Vec<String>, Vec<Stmt>, Span),          // fn(a, b) { ... } — anonymous function value (v0.42 closures)
    CallValue(Box<Expr>, Vec<Expr>, Span),         // <expr>(args) — call a function value (v0.42)
}

/// match pattern.
#[derive(Clone, Debug)]
pub enum Pattern {
    Wildcard,                                  // _
    Bind(String),                              // x  (binds a value)
    Int(i64),
    Bool(bool),
    Str(String),
    Enum(String, String, Vec<Pattern>),        // Enum::Variant(subpats)
    Struct(String, Vec<(String, Pattern)>),    // Name { field: pat, ... }
}

#[derive(Clone, Debug)]
pub struct Arm {
    pub pattern: Pattern,
    pub body: Vec<Stmt>,
}

#[derive(Clone, Debug)]
pub enum Stmt {
    Let {
        name: String,
        ty: Option<String>,
        value: Expr,
        span: Span,
    },
    Assign {
        target: Expr, // Ident or Index — checked as an lvalue by the evaluator
        value: Expr,
        span: Span,
    },
    If {
        // list of (condition, body) pairs = if + elifs. The else body (no condition) is separate.
        branches: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
        span: Span,
    },
    While {
        cond: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    For {
        var: String,
        iter: Expr,
        body: Vec<Stmt>,
        span: Span,
    },
    Fn {
        name: String,
        params: Vec<String>,
        param_types: Vec<Option<String>>, // optional shape annotation per param (parallel to `params`); for the static shape checker
        body: Vec<Stmt>,
        span: Span,
    },
    Return(Option<Expr>, Span),
    Break(Span),
    Continue(Span),
    Import(String, Span), // import "file.wide" — removed by the loader after resolution
    Cout(Vec<Expr>, Span), // cout << e1 << e2 ... — stream output (no auto-space, C++ style)
    Cin(Vec<Expr>, Span),  // cin >> lv1 >> lv2 ... — read whitespace tokens, auto-type, assign (lvalues)
    Struct {
        name: String,
        fields: Vec<String>,
        span: Span,
    },
    Enum {
        name: String,
        variants: Vec<(String, usize)>, // (variant name, arg count)
        span: Span,
    },
    Impl {
        type_name: String,
        methods: Vec<Stmt>, // each a Stmt::Fn — the first parameter is an explicit `self`
        span: Span,
    },
    Match {
        subject: Expr,
        arms: Vec<Arm>,
        span: Span,
    },
    ShowProvenance(Expr, Span), // @show provenance <expr> — introspect the memory provenance record (§3.4, principle 5)
    Trust(Box<Stmt>, Span), // @trust <stmt> — borrow checks off for this statement (WARN, responsibility: caller) (v0.48)
    Expr(Expr),
}

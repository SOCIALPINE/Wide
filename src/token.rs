//! Tokens. v0.2: comparison/logical operators, control-flow keywords, strings, ranges.

use crate::span::Span;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    // literals
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    True,
    False,
    // logical / control-flow keywords
    And,
    Or,
    Not,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Fn,
    Return,
    Break,
    Continue,
    Struct,
    Enum,
    Match,
    Import,
    Impl,
    // operators
    Eq,    // =
    EqEq,  // ==
    NotEq, // !=
    Lt,    // <
    Gt,    // >
    Shl,   // << (cout stream insert)
    Shr,   // >> (cin stream extract)
    Le,    // <=
    Ge,    // >=
    Plus,
    Minus,
    Star,
    Slash,
    Amp, // & — address-of (memory model)
    At,  // @ — directive prefix (@show provenance, memory model introspection)
    // punctuation
    Colon,
    ColonColon, // ::
    FatArrow,   // =>
    Question,   // ?
    Comma,
    Dot,
    DotDot, // ..
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

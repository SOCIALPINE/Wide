//! Recursive descent parser.
//! Expression precedence (low to high): or < and < not < comparison < range < add < mul < unary < postfix.
//! Statements: let · if/elif/else · while · for-in · fn · return · break · continue · expression.

use std::collections::HashSet;

use crate::ast::*;
use crate::span::Span;
use crate::token::{Tok, Token};

pub struct Parser {
    toks: Vec<Token>,
    pos: usize,
    structs: HashSet<String>, // known struct names — resolves `Name { }` literal ambiguity
    pending: Vec<Stmt>,       // extra statements a desugaring produced (e.g. class → struct + impl)
}

impl Parser {
    pub fn new(toks: Vec<Token>, structs: HashSet<String>) -> Self {
        Parser { toks, pos: 0, structs, pending: Vec::new() }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek2(&self) -> &Tok {
        let p = (self.pos + 1).min(self.toks.len() - 1);
        &self.toks[p].tok
    }

    fn peek_span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn advance(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> Result<(), String> {
        if self.peek() == t {
            self.advance();
            Ok(())
        } else {
            Err(format!(
                "line {}: expected '{:?}', found '{:?}'",
                self.peek_span().line,
                t,
                self.peek()
            ))
        }
    }

    fn ident(&mut self) -> Result<String, String> {
        match self.peek().clone() {
            Tok::Ident(n) => {
                self.advance();
                Ok(n)
            }
            other => Err(format!("line {}: expected name, found '{:?}'", self.peek_span().line, other)),
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.advance();
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<Stmt>, String> {
        let stmts = self.parse_stmt_list(&Tok::Eof)?;
        Ok(stmts)
    }

    /// Parse a statement list up to `terminator` (Eof or `}`).
    fn parse_stmt_list(&mut self, terminator: &Tok) -> Result<Vec<Stmt>, String> {
        let mut stmts = Vec::new();
        self.skip_newlines();
        while self.peek() != terminator {
            if matches!(self.peek(), Tok::Eof) {
                return Err(format!("line {}: expected '{:?}', found end of file", self.peek_span().line, terminator));
            }
            stmts.push(self.parse_stmt()?);
            stmts.append(&mut self.pending); // e.g. `class` desugars into struct + impl (v0.54)
            match self.peek() {
                Tok::Newline => self.skip_newlines(),
                t if t == terminator => break,
                other => {
                    return Err(format!(
                        "line {}: unexpected '{:?}' at end of statement",
                        self.peek_span().line,
                        other
                    ))
                }
            }
        }
        Ok(stmts)
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, String> {
        self.eat(&Tok::LBrace)?;
        let body = self.parse_stmt_list(&Tok::RBrace)?;
        self.eat(&Tok::RBrace)?;
        Ok(body)
    }

    fn parse_stmt(&mut self) -> Result<Stmt, String> {
        // cout/cin stream statements — detected by the name followed by `<<` / `>>`.
        let stream = match self.peek() {
            Tok::Ident(n) if n.as_str() == "cout" && matches!(self.peek2(), Tok::Shl) => 1,
            Tok::Ident(n) if n.as_str() == "cin" && matches!(self.peek2(), Tok::Shr) => 2,
            _ => 0,
        };
        if stream == 1 {
            return self.parse_cout();
        }
        if stream == 2 {
            return self.parse_cin();
        }
        match self.peek() {
            Tok::If => self.parse_if(),
            Tok::While => self.parse_while(),
            Tok::For => self.parse_for(),
            Tok::Fn => self.parse_fn(),
            Tok::Struct => self.parse_struct(),
            Tok::Enum => self.parse_enum(),
            Tok::Impl => self.parse_impl(),
            Tok::Class => self.parse_class(),
            Tok::Match => self.parse_match(),
            Tok::Import => self.parse_import(),
            Tok::At => self.parse_directive(),
            Tok::Return => self.parse_return(),
            Tok::Break => {
                let span = self.peek_span();
                self.advance();
                Ok(Stmt::Break(span))
            }
            Tok::Continue => {
                let span = self.peek_span();
                self.advance();
                Ok(Stmt::Continue(span))
            }
            _ => self.parse_let_or_expr(),
        }
    }

    fn parse_let_or_expr(&mut self) -> Result<Stmt, String> {
        // typed declaration: ident : Type = expr
        if let Tok::Ident(name) = self.peek().clone() {
            if matches!(self.peek2(), Tok::Colon) {
                let span = self.peek_span();
                self.advance();
                self.advance();
                let ty = self.parse_type_annot()?;
                self.eat(&Tok::Eq)?;
                let value = self.parse_expr()?;
                return Ok(Stmt::Let { name, ty: Some(ty), value, span });
            }
        }
        // general: parse an expression; if followed by '=', it's an assignment (lvalue = Ident or Index).
        let e = self.parse_expr()?;
        if matches!(self.peek(), Tok::Eq) {
            let span = self.peek_span();
            self.advance();
            let value = self.parse_expr()?;
            return Ok(Stmt::Assign { target: e, value, span });
        }
        Ok(Stmt::Expr(e))
    }

    /// Parse a type annotation into a canonical string. Plain types are a bare ident (`int`); tensor
    /// shape annotations carry a bracketed spec (`tensor[f32, (M, K)]`, `tensor[(?, 768)]`). The string
    /// is interpreted by the static shape checker (§4.1 symbolic tier); eval/compile ignore it.
    fn parse_type_annot(&mut self) -> Result<String, String> {
        let base = self.ident()?;
        if !matches!(self.peek(), Tok::LBracket) {
            return Ok(base);
        }
        let span = self.peek_span();
        self.advance(); // [
        let mut s = String::from("[");
        let mut depth = 1;
        loop {
            match self.peek().clone() {
                Tok::LBracket => {
                    depth += 1;
                    s.push('[');
                }
                Tok::RBracket => {
                    depth -= 1;
                    s.push(']');
                    self.advance();
                    if depth == 0 {
                        break;
                    }
                    continue;
                }
                Tok::LParen => s.push('('),
                Tok::RParen => s.push(')'),
                Tok::Comma => s.push(','),
                Tok::Question => s.push('?'),
                Tok::Ident(n) => s.push_str(&n),
                Tok::Int(n) => s.push_str(&n.to_string()),
                Tok::Eof => return Err(format!("line {}: unterminated type annotation", span.line)),
                other => return Err(format!("line {}: unexpected token in type annotation: {:?}", span.line, other)),
            }
            self.advance();
        }
        Ok(format!("{}{}", base, s))
    }

    fn parse_if(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::If)?;
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        let mut branches = vec![(cond, body)];
        let mut else_body = None;
        loop {
            // `elif`/`else` may appear on the line after `}`. Try skipping newlines,
            // and if it's not elif/else, rewind so the newline remains a statement terminator.
            let save = self.pos;
            self.skip_newlines();
            match self.peek() {
                Tok::Elif => {
                    self.advance();
                    let c = self.parse_expr()?;
                    let b = self.parse_block()?;
                    branches.push((c, b));
                }
                Tok::Else => {
                    self.advance();
                    else_body = Some(self.parse_block()?);
                    break;
                }
                _ => {
                    self.pos = save;
                    break;
                }
            }
        }
        Ok(Stmt::If { branches, else_body, span })
    }

    fn parse_while(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::While)?;
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::While { cond, body, span })
    }

    fn parse_for(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::For)?;
        let var = self.ident()?;
        self.eat(&Tok::In)?;
        let iter = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Stmt::For { var, iter, body, span })
    }

    fn parse_fn(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Fn)?;
        let name = self.ident()?;
        self.eat(&Tok::LParen)?;
        let mut params = Vec::new();
        let mut param_types: Vec<Option<String>> = Vec::new();
        if !matches!(self.peek(), Tok::RParen) {
            loop {
                params.push(self.ident()?);
                // optional shape annotation: `x: tensor[(M, K)]` (used by the static shape checker).
                if matches!(self.peek(), Tok::Colon) {
                    self.advance();
                    param_types.push(Some(self.parse_type_annot()?));
                } else {
                    param_types.push(None);
                }
                if matches!(self.peek(), Tok::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.eat(&Tok::RParen)?;
        let body = self.parse_block()?;
        Ok(Stmt::Fn { name, params, param_types, body, span })
    }

    fn parse_struct(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Struct)?;
        let name = self.ident()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut fields = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            fields.push(self.ident()?);
            // separator (comma) is optional — comma, newline, or whitespace all allowed.
            if matches!(self.peek(), Tok::Comma) {
                self.advance();
            }
            self.skip_newlines();
        }
        self.eat(&Tok::RBrace)?;
        Ok(Stmt::Struct { name, fields, span })
    }

    fn parse_enum(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Enum)?;
        let name = self.ident()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut variants = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            let vname = self.ident()?;
            // optional (arity) — argument names are ignored, only the count matters (positional payload)
            let arity = if matches!(self.peek(), Tok::LParen) {
                self.advance();
                let mut n = 0;
                if !matches!(self.peek(), Tok::RParen) {
                    loop {
                        self.ident()?; // argument name (for documentation), count only
                        n += 1;
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.eat(&Tok::RParen)?;
                n
            } else {
                0
            };
            variants.push((vname, arity));
            // separator (comma) is optional — `A B C`, `A, B`, or newline all allowed.
            if matches!(self.peek(), Tok::Comma) {
                self.advance();
            }
            self.skip_newlines();
        }
        self.eat(&Tok::RBrace)?;
        Ok(Stmt::Enum { name, variants, span })
    }

    fn parse_cout(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.advance(); // cout
        let mut parts = Vec::new();
        while matches!(self.peek(), Tok::Shl) {
            self.advance(); // <<
            parts.push(self.parse_expr()?);
        }
        Ok(Stmt::Cout(parts, span))
    }

    fn parse_cin(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.advance(); // cin
        let mut targets = Vec::new();
        while matches!(self.peek(), Tok::Shr) {
            self.advance(); // >>
            targets.push(self.parse_expr()?); // lvalue: Ident / Index / Field (validated at eval)
        }
        Ok(Stmt::Cin(targets, span))
    }

    fn parse_impl(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Impl)?;
        let type_name = self.ident()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut methods = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            if !matches!(self.peek(), Tok::Fn) {
                return Err(format!(
                    "line {}: only fn allowed in impl block (found: '{:?}')",
                    self.peek_span().line,
                    self.peek()
                ));
            }
            methods.push(self.parse_fn()?);
            self.skip_newlines();
        }
        self.eat(&Tok::RBrace)?;
        Ok(Stmt::Impl { type_name, methods, span })
    }

    /// `class Name { fields... methods... }` — one declaration that desugars into
    /// `struct Name { fields }` + `impl Name { methods }` (v0.54). Fields come first (bare names,
    /// comma/newline separated); every `fn` from the first one on is a method. A method without a
    /// leading `self` parameter is an *associated function*, called as `Name::fn_name(args)`
    /// (the `new` constructor by convention).
    fn parse_class(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Class)?;
        let name = self.ident()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut fields = Vec::new();
        while !matches!(self.peek(), Tok::RBrace | Tok::Fn) {
            fields.push(self.ident()?);
            if matches!(self.peek(), Tok::Comma) {
                self.advance();
            }
            self.skip_newlines();
        }
        let mut methods = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            if !matches!(self.peek(), Tok::Fn) {
                return Err(format!(
                    "line {}: class fields must come before methods (found: '{:?}')",
                    self.peek_span().line,
                    self.peek()
                ));
            }
            methods.push(self.parse_fn()?);
            self.skip_newlines();
        }
        self.eat(&Tok::RBrace)?;
        self.pending.push(Stmt::Impl { type_name: name.clone(), methods, span });
        Ok(Stmt::Struct { name, fields, span })
    }

    fn parse_import(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Import)?;
        match self.peek().clone() {
            Tok::Str(p) => {
                self.advance();
                Ok(Stmt::Import(p, span))
            }
            other => Err(format!("line {}: expected a string path after import, found '{:?}'", span.line, other)),
        }
    }

    /// `@`-directives (memory model introspection, principle 5). Currently only `@show provenance <expr>`.
    fn parse_directive(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::At)?;
        let kw = match self.peek().clone() {
            Tok::Ident(n) => n,
            other => return Err(format!("line {}: expected a directive name after '@', found '{:?}'", span.line, other)),
        };
        if kw == "trust" {
            // @trust <stmt> — borrow checks off for that statement (the trust tier of the gradient).
            self.advance(); // trust
            let inner = self.parse_stmt()?;
            return Ok(Stmt::Trust(Box::new(inner), span));
        }
        if kw != "show" {
            return Err(format!("line {}: unknown directive '@{}' (have @show, @trust)", span.line, kw));
        }
        self.advance(); // show
        let what = match self.peek().clone() {
            Tok::Ident(n) => n,
            other => return Err(format!("line {}: expected what to show after '@show', found '{:?}'", span.line, other)),
        };
        if what != "provenance" {
            return Err(format!("line {}: unknown '@show {}' (have @show provenance)", span.line, what));
        }
        self.advance(); // provenance
        let expr = self.parse_expr()?;
        Ok(Stmt::ShowProvenance(expr, span))
    }

    fn parse_match(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Match)?;
        let subject = self.parse_expr()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut arms = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            let pattern = self.parse_pattern()?;
            self.eat(&Tok::FatArrow)?;
            // body: block `{ }` or a single statement
            let body = if matches!(self.peek(), Tok::LBrace) {
                self.parse_block()?
            } else {
                vec![self.parse_stmt()?]
            };
            arms.push(Arm { pattern, body });
            match self.peek() {
                Tok::Newline => self.skip_newlines(),
                Tok::RBrace => break,
                other => return Err(format!("line {}: expected newline after match arm, found '{:?}'", self.peek_span().line, other)),
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Stmt::Match { subject, arms, span })
    }

    fn parse_pattern(&mut self) -> Result<Pattern, String> {
        match self.peek().clone() {
            Tok::Int(n) => {
                self.advance();
                Ok(Pattern::Int(n))
            }
            Tok::Minus => {
                self.advance();
                match self.peek().clone() {
                    Tok::Int(n) => {
                        self.advance();
                        Ok(Pattern::Int(-n))
                    }
                    _ => Err(format!("line {}: expected integer after '-' in pattern", self.peek_span().line)),
                }
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Pattern::Str(s))
            }
            Tok::True => {
                self.advance();
                Ok(Pattern::Bool(true))
            }
            Tok::False => {
                self.advance();
                Ok(Pattern::Bool(false))
            }
            Tok::Ident(name) => {
                if name == "_" {
                    self.advance();
                    Ok(Pattern::Wildcard)
                } else if matches!(self.peek2(), Tok::ColonColon) {
                    self.advance(); // enum name
                    self.eat(&Tok::ColonColon)?;
                    let variant = self.ident()?;
                    let subpats = if matches!(self.peek(), Tok::LParen) {
                        self.parse_pattern_list()?
                    } else {
                        Vec::new()
                    };
                    Ok(Pattern::Enum(name, variant, subpats))
                } else if matches!(self.peek2(), Tok::LBrace) {
                    self.advance(); // struct name
                    self.eat(&Tok::LBrace)?;
                    let mut fields = Vec::new();
                    if !matches!(self.peek(), Tok::RBrace) {
                        loop {
                            let fname = self.ident()?;
                            let pat = if matches!(self.peek(), Tok::Colon) {
                                self.advance();
                                self.parse_pattern()?
                            } else {
                                Pattern::Bind(fname.clone()) // shorthand: { x } == { x: x }
                            };
                            fields.push((fname, pat));
                            if matches!(self.peek(), Tok::Comma) {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.eat(&Tok::RBrace)?;
                    Ok(Pattern::Struct(name, fields))
                } else {
                    self.advance();
                    Ok(Pattern::Bind(name))
                }
            }
            other => Err(format!("line {}: expected pattern, found '{:?}'", self.peek_span().line, other)),
        }
    }

    fn parse_pattern_list(&mut self) -> Result<Vec<Pattern>, String> {
        self.eat(&Tok::LParen)?;
        let mut pats = Vec::new();
        if !matches!(self.peek(), Tok::RParen) {
            loop {
                pats.push(self.parse_pattern()?);
                if matches!(self.peek(), Tok::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.eat(&Tok::RParen)?;
        Ok(pats)
    }

    fn parse_return(&mut self) -> Result<Stmt, String> {
        let span = self.peek_span();
        self.eat(&Tok::Return)?;
        let val = match self.peek() {
            Tok::Newline | Tok::RBrace | Tok::Eof => None,
            _ => Some(self.parse_expr()?),
        };
        Ok(Stmt::Return(val, span))
    }

    // ---- expressions ----

    fn parse_expr(&mut self) -> Result<Expr, String> {
        self.parse_or()
    }

    fn parse_or(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Tok::Or) {
            let span = self.peek_span();
            self.advance();
            let right = self.parse_and()?;
            left = Expr::Or(Box::new(left), Box::new(right), span);
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Tok::And) {
            let span = self.peek_span();
            self.advance();
            let right = self.parse_not()?;
            left = Expr::And(Box::new(left), Box::new(right), span);
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Tok::Not) {
            let span = self.peek_span();
            self.advance();
            let e = self.parse_not()?;
            return Ok(Expr::Not(Box::new(e), span));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<Expr, String> {
        let left = self.parse_range()?;
        let op = match self.peek() {
            Tok::EqEq => BinOp::Eq,
            Tok::NotEq => BinOp::Ne,
            Tok::Lt => BinOp::Lt,
            Tok::Gt => BinOp::Gt,
            Tok::Le => BinOp::Le,
            Tok::Ge => BinOp::Ge,
            _ => return Ok(left),
        };
        let span = self.peek_span();
        self.advance();
        let right = self.parse_range()?;
        Ok(Expr::Binary(op, Box::new(left), Box::new(right), span))
    }

    fn parse_range(&mut self) -> Result<Expr, String> {
        let left = self.parse_add()?;
        if matches!(self.peek(), Tok::DotDot) {
            let span = self.peek_span();
            self.advance();
            let right = self.parse_add()?;
            return Ok(Expr::Range(Box::new(left), Box::new(right), span));
        }
        Ok(left)
    }

    fn parse_add(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_mul()?;
        loop {
            let op = match self.peek() {
                Tok::Plus => BinOp::Add,
                Tok::Minus => BinOp::Sub,
                _ => break,
            };
            let span = self.peek_span();
            self.advance();
            let right = self.parse_mul()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), span);
        }
        Ok(left)
    }

    fn parse_mul(&mut self) -> Result<Expr, String> {
        let mut left = self.parse_unary()?;
        loop {
            let op = match self.peek() {
                Tok::Star => BinOp::Mul,
                Tok::Slash => BinOp::Div,
                _ => break,
            };
            let span = self.peek_span();
            self.advance();
            let right = self.parse_unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right), span);
        }
        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr, String> {
        if matches!(self.peek(), Tok::Minus) {
            let span = self.peek_span();
            self.advance();
            let e = self.parse_unary()?;
            return Ok(Expr::Neg(Box::new(e), span));
        }
        if matches!(self.peek(), Tok::Amp) {
            let span = self.peek_span();
            self.advance();
            // `&mut x` — exclusive borrow (v0.48). `mut` is not a keyword: it is recognised here only
            // when an expression follows it (so a variable literally named `mut` still works as `&mut`).
            if matches!(self.peek(), Tok::Ident(n) if n == "mut") && matches!(self.peek2(), Tok::Ident(_)) {
                self.advance(); // mut
                let e = self.parse_unary()?;
                return Ok(Expr::AddrOfMut(Box::new(e), span));
            }
            let e = self.parse_unary()?;
            return Ok(Expr::AddrOf(Box::new(e), span));
        }
        if matches!(self.peek(), Tok::Star) {
            // prefix `*` = dereference (infix `*` multiplication is handled in parse_mul)
            let span = self.peek_span();
            self.advance();
            let e = self.parse_unary()?;
            return Ok(Expr::Deref(Box::new(e), span));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr, String> {
        let mut e = self.parse_primary()?;
        loop {
            match self.peek() {
                Tok::Dot => {
                    let span = self.peek_span();
                    self.advance();
                    let name = self.ident()?;
                    if matches!(self.peek(), Tok::LParen) {
                        let args = self.parse_args()?;
                        e = Expr::Method(Box::new(e), name, args, span);
                    } else {
                        e = Expr::Field(Box::new(e), name, span);
                    }
                }
                Tok::LParen => {
                    let span = self.peek_span();
                    if let Expr::Ident(name, _) = e {
                        let args = self.parse_args()?;
                        e = Expr::Call(name, args, span);
                    } else {
                        // calling any other expression calls it as a function *value* (closures, v0.42)
                        let args = self.parse_args()?;
                        e = Expr::CallValue(Box::new(e), args, span);
                    }
                }
                Tok::LBracket => {
                    let span = self.peek_span();
                    self.advance();
                    let idx = self.parse_expr()?;
                    self.eat(&Tok::RBracket)?;
                    e = Expr::Index(Box::new(e), Box::new(idx), span);
                }
                Tok::Question => {
                    let span = self.peek_span();
                    self.advance();
                    e = Expr::Try(Box::new(e), span);
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_args(&mut self) -> Result<Vec<Expr>, String> {
        self.eat(&Tok::LParen)?;
        let mut args = Vec::new();
        if !matches!(self.peek(), Tok::RParen) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.peek(), Tok::Comma) {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.eat(&Tok::RParen)?;
        Ok(args)
    }

    fn parse_match_expr(&mut self) -> Result<Expr, String> {
        let span = self.peek_span();
        self.eat(&Tok::Match)?;
        let subject = self.parse_expr()?;
        self.eat(&Tok::LBrace)?;
        self.skip_newlines();
        let mut arms = Vec::new();
        while !matches!(self.peek(), Tok::RBrace) {
            let pat = self.parse_pattern()?;
            self.eat(&Tok::FatArrow)?;
            if matches!(self.peek(), Tok::LBrace) {
                return Err(format!(
                    "line {}: match-expression arms must be expressions (use statement match for blocks)",
                    self.peek_span().line
                ));
            }
            let body = self.parse_expr()?;
            arms.push((pat, body));
            match self.peek() {
                Tok::Newline => self.skip_newlines(),
                Tok::Comma => {
                    self.advance();
                    self.skip_newlines();
                }
                Tok::RBrace => break,
                other => {
                    return Err(format!(
                        "line {}: expected newline/comma after match-expression arm, found '{:?}'",
                        self.peek_span().line, other
                    ))
                }
            }
        }
        self.eat(&Tok::RBrace)?;
        Ok(Expr::Match(Box::new(subject), arms, span))
    }

    fn parse_primary(&mut self) -> Result<Expr, String> {
        let span = self.peek_span();
        match self.peek().clone() {
            Tok::Match => self.parse_match_expr(),
            Tok::Fn => {
                // anonymous function value: fn(a, b) { ... } (v0.42 closures)
                self.advance();
                self.eat(&Tok::LParen)?;
                let mut params = Vec::new();
                if !matches!(self.peek(), Tok::RParen) {
                    loop {
                        params.push(self.ident()?);
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.eat(&Tok::RParen)?;
                let body = self.parse_block()?;
                Ok(Expr::Lambda(params, body, span))
            }
            Tok::Int(n) => {
                self.advance();
                Ok(Expr::Int(n, span))
            }
            Tok::Float(f) => {
                self.advance();
                Ok(Expr::Float(f, span))
            }
            Tok::Str(s) => {
                self.advance();
                Ok(Expr::Str(s, span))
            }
            Tok::True => {
                self.advance();
                Ok(Expr::Bool(true, span))
            }
            Tok::False => {
                self.advance();
                Ok(Expr::Bool(false, span))
            }
            Tok::Ident(name) => {
                if matches!(self.peek2(), Tok::ColonColon) {
                    // enum construction: Enum::Variant or Enum::Variant(args)
                    self.advance();
                    self.eat(&Tok::ColonColon)?;
                    let variant = self.ident()?;
                    let args = if matches!(self.peek(), Tok::LParen) {
                        self.parse_args()?
                    } else {
                        Vec::new()
                    };
                    Ok(Expr::EnumLit(name, variant, args, span))
                } else if name == "map" && matches!(self.peek2(), Tok::LBrace) {
                    self.advance();
                    self.eat(&Tok::LBrace)?;
                    self.eat(&Tok::RBrace)?;
                    Ok(Expr::Map(span))
                } else if self.structs.contains(&name) && matches!(self.peek2(), Tok::LBrace) {
                    // struct literal: Name { field: expr, ... }  (shorthand { x } == { x: x })
                    self.advance();
                    self.eat(&Tok::LBrace)?;
                    self.skip_newlines();
                    let mut fields = Vec::new();
                    if !matches!(self.peek(), Tok::RBrace) {
                        loop {
                            let fname = self.ident()?;
                            let val = if matches!(self.peek(), Tok::Colon) {
                                self.advance();
                                self.parse_expr()?
                            } else {
                                Expr::Ident(fname.clone(), span)
                            };
                            fields.push((fname, val));
                            if matches!(self.peek(), Tok::Comma) {
                                self.advance();
                                self.skip_newlines();
                            } else {
                                break;
                            }
                        }
                    }
                    self.skip_newlines();
                    self.eat(&Tok::RBrace)?;
                    Ok(Expr::StructLit(name, fields, span))
                } else {
                    self.advance();
                    Ok(Expr::Ident(name, span))
                }
            }
            Tok::LParen => {
                self.advance();
                let e = self.parse_expr()?;
                self.eat(&Tok::RParen)?;
                Ok(e)
            }
            Tok::LBracket => {
                self.advance();
                let mut elems = Vec::new();
                if !matches!(self.peek(), Tok::RBracket) {
                    loop {
                        elems.push(self.parse_expr()?);
                        if matches!(self.peek(), Tok::Comma) {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                }
                self.eat(&Tok::RBracket)?;
                Ok(Expr::Array(elems, span))
            }
            other => Err(format!("line {}: unexpected '{:?}'", span.line, other)),
        }
    }
}

//! Lexer. Source → tokens. `#` comments; newlines are preserved as statement separators —
//! EXCEPT inside `(...)` / `[...]` (v0.45): there a newline is just whitespace, so multi-line
//! array literals and argument lists work. `{...}` keeps newlines (blocks need them as separators).
//! Multi-char: `==` `!=` `<=` `>=` `..`. Strings `"..."` (escapes \n \t \\ \").

use crate::span::Span;
use crate::token::{Tok, Token};

pub fn lex(src: &str) -> Result<Vec<Token>, String> {
    let chars: Vec<char> = src.chars().collect();
    let mut i = 0;
    let mut line = 1;
    let mut col = 1;
    let mut out = Vec::new();
    // Nesting stack (v0.45): `true` = ( or [ (newlines skipped), `false` = { (newlines kept — blocks
    // need them as statement separators, even when the block sits inside parens, e.g. a lambda arg).
    let mut nest: Vec<bool> = Vec::new();

    while i < chars.len() {
        let c = chars[i];
        let start = Span::new(line, col);
        let next = chars.get(i + 1).copied();

        // single/double-char token helper: (token, consumed length)
        let single = |t: Tok| Some((t, 1usize));
        let emitted: Option<(Tok, usize)> = match c {
            ' ' | '\t' | '\r' => {
                i += 1;
                col += 1;
                continue;
            }
            '\n' => {
                if nest.last() != Some(&true) {
                    out.push(Token { tok: Tok::Newline, span: start });
                }
                i += 1;
                line += 1;
                col = 1;
                continue;
            }
            '#' => {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                    col += 1;
                }
                continue;
            }
            '"' => {
                let (s, consumed) = lex_string(&chars, i, line)?;
                out.push(Token { tok: Tok::Str(s), span: start });
                i += consumed;
                col += consumed;
                continue;
            }
            '=' if next == Some('=') => Some((Tok::EqEq, 2)),
            '=' if next == Some('>') => Some((Tok::FatArrow, 2)),
            '=' => single(Tok::Eq),
            ':' if next == Some(':') => Some((Tok::ColonColon, 2)),
            '!' if next == Some('=') => Some((Tok::NotEq, 2)),
            '<' if next == Some('<') => Some((Tok::Shl, 2)),
            '<' if next == Some('=') => Some((Tok::Le, 2)),
            '<' => single(Tok::Lt),
            '>' if next == Some('>') => Some((Tok::Shr, 2)),
            '>' if next == Some('=') => Some((Tok::Ge, 2)),
            '>' => single(Tok::Gt),
            '.' if next == Some('.') => Some((Tok::DotDot, 2)),
            '.' => single(Tok::Dot),
            ':' => single(Tok::Colon),
            '&' => single(Tok::Amp),
            '@' => single(Tok::At),
            '+' => single(Tok::Plus),
            '-' => single(Tok::Minus),
            '*' => single(Tok::Star),
            '/' => single(Tok::Slash),
            '(' | '[' => {
                nest.push(true);
                single(if c == '(' { Tok::LParen } else { Tok::LBracket })
            }
            ')' | ']' => {
                if nest.last() == Some(&true) {
                    nest.pop();
                }
                single(if c == ')' { Tok::RParen } else { Tok::RBracket })
            }
            '{' => {
                nest.push(false);
                single(Tok::LBrace)
            }
            '}' => {
                if nest.last() == Some(&false) {
                    nest.pop();
                }
                single(Tok::RBrace)
            }
            ',' => single(Tok::Comma),
            '?' => single(Tok::Question),
            c if c.is_ascii_digit() => {
                let (tok, consumed) = lex_number(&chars, i, line)?;
                out.push(Token { tok, span: start });
                i += consumed;
                col += consumed;
                continue;
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut s = String::new();
                while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    s.push(chars[i]);
                    i += 1;
                    col += 1;
                }
                out.push(Token { tok: keyword(&s), span: start });
                continue;
            }
            other => return Err(format!("line {}: unexpected character '{}'", line, other)),
        };

        if let Some((tok, len)) = emitted {
            out.push(Token { tok, span: start });
            i += len;
            col += len;
        }
    }

    out.push(Token { tok: Tok::Eof, span: Span::new(line, col) });
    Ok(out)
}

fn keyword(s: &str) -> Tok {
    match s {
        "true" => Tok::True,
        "false" => Tok::False,
        "and" => Tok::And,
        "or" => Tok::Or,
        "not" => Tok::Not,
        "if" => Tok::If,
        "elif" => Tok::Elif,
        "else" => Tok::Else,
        "while" => Tok::While,
        "for" => Tok::For,
        "in" => Tok::In,
        "fn" => Tok::Fn,
        "return" => Tok::Return,
        "break" => Tok::Break,
        "continue" => Tok::Continue,
        "struct" => Tok::Struct,
        "enum" => Tok::Enum,
        "match" => Tok::Match,
        "import" => Tok::Import,
        "impl" => Tok::Impl,
        "class" => Tok::Class,
        _ => Tok::Ident(s.to_string()),
    }
}

fn lex_number(chars: &[char], start: usize, line: usize) -> Result<(Tok, usize), String> {
    let mut j = start;
    let mut s = String::new();
    while j < chars.len() && chars[j].is_ascii_digit() {
        s.push(chars[j]);
        j += 1;
    }
    // float? only when `.` is followed by a digit (excludes the `..` in `0..n`).
    if j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
        s.push('.');
        j += 1;
        while j < chars.len() && chars[j].is_ascii_digit() {
            s.push(chars[j]);
            j += 1;
        }
        let f: f64 = s.parse().map_err(|_| format!("line {}: invalid float '{}'", line, s))?;
        Ok((Tok::Float(f), j - start))
    } else {
        let n: i64 = s.parse().map_err(|_| format!("line {}: invalid integer '{}'", line, s))?;
        Ok((Tok::Int(n), j - start))
    }
}

fn lex_string(chars: &[char], start: usize, line: usize) -> Result<(String, usize), String> {
    // chars[start] == '"'
    let mut j = start + 1;
    let mut s = String::new();
    while j < chars.len() {
        match chars[j] {
            '"' => return Ok((s, j - start + 1)),
            '\\' => {
                j += 1;
                match chars.get(j) {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('\\') => s.push('\\'),
                    Some('"') => s.push('"'),
                    Some(other) => return Err(format!("line {}: unknown escape '\\{}'", line, other)),
                    None => return Err(format!("line {}: unterminated string", line)),
                }
                j += 1;
            }
            '\n' => return Err(format!("line {}: newline inside string", line)),
            c => {
                s.push(c);
                j += 1;
            }
        }
    }
    Err(format!("line {}: unterminated string (no closing \")", line))
}

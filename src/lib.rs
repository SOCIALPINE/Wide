//! wide v0.2 — library crate. Exposes the modules so tests and the CLI can share them.

// v0.x scaffolding: literal spans and type annotations aren't read yet, but we
// carry them on the nodes from now on, for the next version's literal-level
// illumination and type checking (the illumination-channel first-class principle).
#![allow(dead_code)]

pub mod ast;
pub mod bytecode;
pub mod check;
pub mod compile;
pub mod eval;
#[cfg(feature = "gpu")]
pub mod gpu;
#[cfg(feature = "jit")]
pub mod jit;
pub mod lexer;
pub mod lumen;
pub mod parser;
pub mod runtime;
pub mod span;
pub mod token;
pub mod value;
pub mod vm;

pub use eval::Interp;
pub use value::Value;
pub use vm::Vm;

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use token::{Tok, Token};

/// Collect `struct Name` / `class Name` names from the tokens — to tell `Name { }` literals apart from blocks.
fn scan_struct_names(toks: &[Token], out: &mut HashSet<String>) {
    for w in toks.windows(2) {
        if matches!(w[0].tok, Tok::Struct | Tok::Class) {
            if let Tok::Ident(n) = &w[1].tok {
                out.insert(n.clone());
            }
        }
    }
}

/// Extract the `import "..."` paths from the tokens.
fn scan_imports(toks: &[Token]) -> Vec<String> {
    let mut v = Vec::new();
    for w in toks.windows(2) {
        if matches!(w[0].tok, Tok::Import) {
            if let Tok::Str(p) = &w[1].tok {
                if !p.starts_with("std/") {
                    v.push(p.clone()); // std/ modules are built-in, not files — don't load a file
                }
            }
        }
    }
    v
}

/// Standard-library modules *written in wide itself* (embedded in the binary; dogfooding).
/// `import "std/ml"` splices these definitions in at the import site.
const STD_ML_SRC: &str = include_str!("stdlib/ml.wide");

/// Expand std modules that carry wide source: the first `import "std/<m>"` with an embedded source is
/// followed by that source's statements (the import marker itself stays — the evaluator's gating reads
/// it). Duplicated imports splice once.
fn expand_std_sources(stmts: Vec<ast::Stmt>) -> Result<Vec<ast::Stmt>, String> {
    let mut out = Vec::with_capacity(stmts.len());
    let mut done = false;
    for s in stmts {
        let is_ml = matches!(&s, ast::Stmt::Import(p, _) if p == "std/ml");
        out.push(s);
        if is_ml && !done {
            done = true;
            let toks = lexer::lex(STD_ML_SRC).map_err(|e| format!("(std/ml) {}", e))?;
            let mut structs = HashSet::new();
            scan_struct_names(&toks, &mut structs);
            let mut p = parser::Parser::new(toks, structs);
            let ml = p.parse_program().map_err(|e| format!("(std/ml) {}", e))?;
            out.extend(ml);
        }
    }
    Ok(out)
}

/// Parse source tokens→AST (single file).
pub fn parse(source: &str) -> Result<Vec<ast::Stmt>, String> {
    let toks = lexer::lex(source)?;
    let mut structs = HashSet::new();
    scan_struct_names(&toks, &mut structs);
    let mut p = parser::Parser::new(toks, structs);
    expand_std_sources(p.parse_program()?)
}

/// Static-check errors as strings (for tests).
pub fn type_errors(source: &str) -> Result<Vec<String>, String> {
    let prog = parse(source)?;
    Ok(check::check(&prog)
        .into_iter()
        .map(|e| format!("line {}: {}", e.line, e.msg))
        .collect())
}

/// Run the source and return the Interp (holding env and illumination). A test convenience.
/// For convenience it auto-enables the std modules (ai/heap/set). (The real CLI requires explicit imports — the
/// gating is enforced by the static checks `type_errors`/`check`.)
pub fn eval_program(source: &str) -> Result<Interp, String> {
    let prelude = "import \"std/ai\"\nimport \"std/heap\"\nimport \"std/set\"\n";
    let prog = parse(&format!("{}{}", prelude, source))?;
    let mut interp = Interp::new();
    interp.run(&prog)?;
    Ok(interp)
}

/// Compile a program to bytecode (stage 2). Errors name any construct the VM doesn't support yet.
pub fn compile_program(source: &str) -> Result<bytecode::Program, String> {
    let prog = parse(source)?;
    compile::compile(&prog)
}

/// Parse → compile to bytecode → run on the VM. Returns the Vm (env + illumination). A test convenience.
/// No std prelude (the VM core subset has no tensors). The tree-walker (`eval_program`) remains the reference.
pub fn eval_program_vm(source: &str) -> Result<vm::Vm, String> {
    let compiled = compile_program(source)?;
    let mut machine = vm::Vm::new();
    machine.run(&compiled)?;
    Ok(machine)
}

/// Like `eval_program_vm` but feeds `cin` from `input` (for tests).
pub fn eval_program_vm_with_input(source: &str, input: &str) -> Result<vm::Vm, String> {
    let compiled = compile_program(source)?;
    let mut machine = vm::Vm::new();
    machine.set_input(input);
    machine.run(&compiled)?;
    Ok(machine)
}

/// Like `eval_program` but feeds `cin` from `input` (whitespace-split) instead of stdin. (For tests / embedding.)
pub fn eval_program_with_input(source: &str, input: &str) -> Result<Interp, String> {
    let prelude = "import \"std/ai\"\nimport \"std/heap\"\nimport \"std/set\"\n";
    let prog = parse(&format!("{}{}", prelude, source))?;
    let mut interp = Interp::new();
    interp.set_input(input);
    interp.run(&prog)?;
    Ok(interp)
}

/// Load the entry file and recursively resolve `import`s into a *flat* list of statements.
///
/// 3 stages: ① collect the graph in dependency order (canonical paths block duplicates and cycles),
/// ② gather every file's struct names *globally* (so cross-file `Name { }` literals are recognized),
/// ③ parse each file with that global set, strip the import statements, and concatenate.
/// Dependencies come first, so combined with hoisting, cross-file fn/struct/enum references work.
pub fn load_file(path: &Path) -> Result<Vec<ast::Stmt>, String> {
    let mut visited = HashSet::new();
    let mut files: Vec<(PathBuf, Vec<Token>)> = Vec::new();
    gather(path, &mut visited, &mut files)?;

    let mut structs = HashSet::new();
    for (_, toks) in &files {
        scan_struct_names(toks, &mut structs);
    }

    let mut out = Vec::new();
    for (canon, toks) in &files {
        let mut p = parser::Parser::new(toks.clone(), structs.clone());
        let stmts = p
            .parse_program()
            .map_err(|e| format!("({}) {}", canon.display(), e))?;
        for s in stmts {
            match &s {
                // file imports are already resolved → remove. std/ imports are kept (the evaluator enables the module).
                ast::Stmt::Import(p, _) if !p.starts_with("std/") => {}
                _ => out.push(s),
            }
        }
    }
    expand_std_sources(out)
}

/// Collect the import graph in dependency-first order. Pushes (canon, tokens) onto files.
fn gather(path: &Path, visited: &mut HashSet<PathBuf>, files: &mut Vec<(PathBuf, Vec<Token>)>) -> Result<(), String> {
    let canon = path
        .canonicalize()
        .map_err(|e| format!("import path '{}': {}", path.display(), e))?;
    if !visited.insert(canon.clone()) {
        return Ok(()); // already collected — blocks duplicates and cycles
    }
    let source =
        std::fs::read_to_string(&canon).map_err(|e| format!("failed to read '{}': {}", canon.display(), e))?;
    let toks = lexer::lex(&source).map_err(|e| format!("({}) {}", canon.display(), e))?;
    let base = canon.parent().map(|p| p.to_path_buf()).unwrap_or_default();
    for imp in scan_imports(&toks) {
        gather(&base.join(&imp), visited, files)
            .map_err(|e| format!("(import \"{}\" from {}) {}", imp, canon.display(), e))?;
    }
    files.push((canon, toks));
    Ok(())
}

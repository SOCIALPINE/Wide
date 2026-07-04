//! wide v0.4 — CLI. Takes a file, runs it, and shows the program output and *illumination* separately.
//! The key demo is the illumination report woven into the source lines — it shows the cost other languages hide.

use std::fs;
use std::path::Path;

use wide::lumen::{Level, Lumen};
use wide::Interp;

const DEMO: &str = r#"# wide v0.4 — control-flow and function demo
fn fib(n) {
    if n < 2 { return n }
    return fib(n - 1) + fib(n - 2)
}

print(fib(10))

total = 0
for i in 1..5 {
    total = total + i
}
print(total)

xs = [3, 1, 4, 1, 5]
print(xs.sum())
"#;

const HELP: &str = "wide — illuminate, don't isolate

A language that shows costs and risks instead of hiding them: every run prints
your program's output followed by an illumination report (INFO:/WARN: lines
woven into the source).

USAGE:
    wide [OPTIONS] [FILE]

ARGUMENTS:
    [FILE]          a .wide program to run (omitted: runs a small built-in demo)

OPTIONS:
    --vm            run on the bytecode VM backend (default: tree-walker)
    --time          print the execution time to stderr (excludes process startup)
    --no-illum      suppress the illumination report
    -h, --help      show this help
    -V, --version   show the version

STREAMS (grader/pipeline friendly):
    stdout          the program's own output (print/cout) — nothing else
    stderr          the banner, the illumination report (INFO:/WARN:), errors, --time

The run sequence is always: static check -> run -> illumination report.
Start with GUIDE.en.md (English) or GUIDE.ko.md (Korean); examples live in examples/.";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", HELP);
        return;
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("wide {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    // `--vm`: run via the bytecode compiler + VM (stage 2). Default is the tree-walker (the reference).
    let use_vm = args.iter().any(|a| a == "--vm");
    // `--time`: print the execution wall time to stderr (excludes process startup — for benchmarks;
    // the OS can add hundreds of ms of process-launch overhead, e.g. antivirus exe scanning).
    let time_it = args.iter().any(|a| a == "--time");
    let t0 = std::time::Instant::now();
    let file = args.iter().skip(1).find(|a| !a.starts_with("--"));
    let result = match (use_vm, file) {
        (true, Some(path)) => run_file_vm(path),
        (true, None) => run_source_vm(DEMO, "<demo>"),
        (false, Some(path)) => run_file(&path.clone()),
        (false, None) => run_source(DEMO, "<demo>"),
    };
    if time_it {
        eprintln!("time: {:.3} ms", t0.elapsed().as_secs_f64() * 1000.0);
    }
    if let Err(e) = result {
        eprintln!("\x1b[31merror:\x1b[0m {}", e);
        std::process::exit(1);
    }
}

/// Run a file through the bytecode VM (stage 2). Resolves imports via load_file (module flattening).
fn run_file_vm(path: &str) -> Result<(), String> {
    let prog = wide::load_file(Path::new(path))?;
    let source = fs::read_to_string(path).unwrap_or_default();
    run_prog_vm(&prog, &source, path)
}

fn run_source_vm(source: &str, name: &str) -> Result<(), String> {
    let prog = wide::parse(source)?;
    run_prog_vm(&prog, source, name)
}

fn run_prog_vm(prog: &[wide::ast::Stmt], source: &str, name: &str) -> Result<(), String> {
    let errors = wide::check::check(prog);
    if !errors.is_empty() {
        eprintln!("\x1b[31m{} type error(s):\x1b[0m", errors.len());
        for e in &errors {
            eprintln!("  \x1b[33mline {}\x1b[0m: {}", e.line, e.msg);
        }
        return Err(format!("static check failed ({} errors) — not running", errors.len()));
    }
    let compiled = wide::compile::compile(prog)?;
    // stdout carries ONLY the program's own output (grader/pipeline friendly, v0.56) — meta → stderr.
    eprintln!("=== wide v{} (vm) · {} ===", env!("CARGO_PKG_VERSION"), name);
    let mut machine = wide::Vm::new();
    let runtime = machine.run(&compiled);
    report_illumination(source, &machine.channel);
    runtime
}

/// File entry — runs the flat program after resolving imports (load_file).
fn run_file(path: &str) -> Result<(), String> {
    let prog = wide::load_file(Path::new(path))?;
    // the illumination report is based on the entry file's source (line mapping for imported modules is best-effort).
    let source = fs::read_to_string(path).unwrap_or_default();
    run_prog(&prog, &source, path)
}

fn run_source(source: &str, name: &str) -> Result<(), String> {
    let prog = wide::parse(source)?;
    run_prog(&prog, source, name)
}

fn run_prog(prog: &[wide::ast::Stmt], source: &str, name: &str) -> Result<(), String> {
    // static type check — *before* running, gather and report name/arity errors (§4.5 safety gradient).
    let errors = wide::check::check(prog);
    if !errors.is_empty() {
        eprintln!("\x1b[31m{} type error(s):\x1b[0m", errors.len());
        let lines: Vec<&str> = source.lines().collect();
        for e in &errors {
            let src = lines.get(e.line - 1).copied().unwrap_or("").trim();
            eprintln!("  \x1b[33mline {}\x1b[0m: {}   │ {}", e.line, e.msg, src);
        }
        return Err(format!("static check failed ({} errors) — not running", errors.len()));
    }

    // stdout carries ONLY the program's own output (grader/pipeline friendly, v0.56) — meta → stderr.
    eprintln!("=== wide v{} · {} ===", env!("CARGO_PKG_VERSION"), name);

    let mut interp = Interp::new();
    let runtime = interp.run(prog);

    // illumination is *always* shown even on error — often that WARN: diagnoses the error (principle 1).
    report_illumination(source, &interp.channel);
    runtime
}

/// Illumination report — weaves INFO:/WARN: labels into the source lines.
fn report_illumination(source: &str, ch: &wide::lumen::Channel) {
    if std::env::args().any(|a| a == "--no-illum") {
        return;
    }
    let recs = &ch.records;
    if recs.is_empty() {
        return;
    }
    eprintln!("\n— illumination (visible-cost) —");
    let lines: Vec<&str> = source.lines().collect();

    let mut by_line: std::collections::BTreeMap<usize, Vec<&Lumen>> = Default::default();
    for r in recs {
        by_line.entry(r.span.line).or_default().push(r);
    }

    for (line, rs) in by_line {
        let src = lines.get(line - 1).copied().unwrap_or("").trim_end();
        eprintln!("{:>3} | {}", line, src);
        for r in rs {
            let (label, color) = match r.level {
                Level::Info => ("INFO", "\x1b[36m"),
                Level::Warn => ("WARN", "\x1b[33m"),
            };
            if r.count > 1 {
                eprintln!("    {}  {}: {} (× {})\x1b[0m", color, label, r.msg, r.count);
            } else {
                eprintln!("    {}  {}: {}\x1b[0m", color, label, r.msg);
            }
        }
    }
    if ch.truncated > 0 {
        eprintln!("    \x1b[33m  WARN: illumination truncated — {} further unique records were not stored\x1b[0m", ch.truncated);
    }
}

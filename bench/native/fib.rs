// bench: recursive fib(30) — Rust reference (self-timed like `wide --time`).
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    let t0 = std::time::Instant::now();
    let r = fib(30);
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("{}", r);
    eprintln!("time: {:.3} ms", ms);
}

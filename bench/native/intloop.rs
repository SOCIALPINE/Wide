// bench: 10M-iteration integer while loop — Rust reference (self-timed).
fn main() {
    let t0 = std::time::Instant::now();
    let mut total: i64 = 0;
    let mut i: i64 = 0;
    while i < 10_000_000 {
        // black_box each iteration: without it LLVM folds the whole loop into the closed-form
        // sum (measured 0 ms) — the C version prevents the same folding with `volatile`.
        total = std::hint::black_box(total + i);
        i += 1;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("{}", total);
    eprintln!("time: {:.3} ms", ms);
}

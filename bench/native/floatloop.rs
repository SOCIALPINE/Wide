// bench: 10M-iteration f64 while loop — Rust reference (self-timed).
fn main() {
    let t0 = std::time::Instant::now();
    let mut total: f64 = 0.0;
    let mut i: f64 = 0.0;
    while i < 10_000_000.0 {
        total = std::hint::black_box(total + i); // match the C version's volatile accumulator
        i += 1.0;
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("{:.0}", total);
    eprintln!("time: {:.3} ms", ms);
}

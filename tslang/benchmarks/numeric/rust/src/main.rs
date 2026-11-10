// Rust port of tslang/benchmarks/numeric/nbench.ts: the same six kernels over Vec<f64> / Vec<i32>,
// written the way a Rust programmer would (slices and iterators where they fit), printing the same
// six check values.

const N: usize = 4096;
const M: usize = 64;

fn check(v: f64) -> f64 { (v * 1000.0).floor() % 999983.0 }

fn make_array(n: usize, mul: f64, modulus: usize, add: f64) -> Vec<f64> {
    (0..n).map(|i| (i % modulus) as f64 * mul + add).collect()
}

fn sum(a: &[f64]) -> f64 {
    let mut s = 0.0;
    for v in a { s += *v; }
    s
}

#[inline(never)]
fn saxpy(a: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x) { *yi += a * *xi; }
}

#[inline(never)]
fn triad(x: &[f64], y: &[f64], z: &mut [f64], c: f64) {
    for ((zi, xi), yi) in z.iter_mut().zip(x).zip(y) { *zi = *xi * *yi + c; }
}

#[inline(never)]
fn stencil(src: &[f64], dst: &mut [f64]) {
    let n = src.len();
    for (d, w) in dst[1..n - 1].iter_mut().zip(src.windows(3)) {
        *d = (w[0] + 2.0 * w[1] + w[2]) * 0.25;
    }
}

#[inline(never)]
fn dot(x: &[f64], y: &[f64]) -> f64 {
    let mut s = 0.0;
    for (xi, yi) in x.iter().zip(y) { s += *xi * *yi; }
    s
}

#[inline(never)]
fn matmul(a: &[f64], b: &[f64], c: &mut [f64], m: usize) {
    for i in 0..m {
        for k in 0..m {
            let aik = a[i * m + k];
            let row = &b[k * m..k * m + m];
            for (cj, bj) in c[i * m..i * m + m].iter_mut().zip(row) { *cj += aik * *bj; }
        }
    }
}

#[inline(never)]
fn hash(h: &mut [i32], g: &[i32]) -> i32 {
    let mut s: i32 = 0;
    for (hi, gi) in h.iter_mut().zip(g) {
        *hi = (hi.wrapping_mul(31).wrapping_add(*gi)) & 0xffff;
        s = s.wrapping_add(*hi);
    }
    s
}

fn run_saxpy() -> f64 {
    let x = make_array(N, 0.5, 97, 1.0);
    let mut y = make_array(N, 0.25, 89, 2.0);
    for _ in 0..60000 { saxpy(0.000001, &x, &mut y); }
    check(sum(&y))
}

fn run_triad() -> f64 {
    let x = make_array(N, 0.5, 97, 1.0);
    let y = make_array(N, 0.25, 89, 2.0);
    let mut z = make_array(N, 0.0, 1, 0.0);
    let mut s = 0.0;
    for r in 0..60000usize {
        triad(&x, &y, &mut z, r as f64 * 0.001);
        s += z[r % N];
    }
    check(s + sum(&z))
}

fn run_stencil() -> f64 {
    let mut a = make_array(N, 0.5, 97, 1.0);
    let mut b = make_array(N, 0.5, 97, 1.0);
    for _ in 0..30000 {
        stencil(&a, &mut b);
        stencil(&b, &mut a);
    }
    check(sum(&a))
}

fn run_dot() -> f64 {
    let mut x = make_array(N, 0.5, 97, 1.0);
    let y = make_array(N, 0.25, 89, 2.0);
    let mut s = 0.0;
    for r in 0..30000usize {
        x[r % N] += 0.5;
        s += dot(&x, &y) * 0.000001;
    }
    check(s)
}

fn run_matmul() -> f64 {
    let a = make_array(M * M, 0.001, 13, 0.01);
    let b = make_array(M * M, 0.002, 7, 0.02);
    let mut c = make_array(M * M, 0.0, 1, 0.0);
    for _ in 0..400 { matmul(&a, &b, &mut c, M); }
    check(sum(&c))
}

fn run_hash() -> i32 {
    let mut h: Vec<i32> = (0..N as i32).map(|i| i % 251).collect();
    let g: Vec<i32> = (0..N as i32).map(|i| i % 241 + 7).collect();
    let mut s: i32 = 0;
    for _ in 0..60000 { s = s.wrapping_add(hash(&mut h, &g)) & 0xfffff; }
    s
}

// With no argument every kernel runs; with a kernel's name only that one does.
fn main() {
    let only = std::env::args().nth(1);
    let run = |name: &str| only.as_deref().map_or(true, |o| o == name);
    println!("start...");
    if run("saxpy") { println!("saxpy: {}", run_saxpy()); }
    if run("triad") { println!("triad: {}", run_triad()); }
    if run("stencil") { println!("stencil: {}", run_stencil()); }
    if run("dot") { println!("dot: {}", run_dot()); }
    if run("matmul") { println!("matmul: {}", run_matmul()); }
    if run("hash") { println!("hash: {}", run_hash()); }
    println!("done.");
}

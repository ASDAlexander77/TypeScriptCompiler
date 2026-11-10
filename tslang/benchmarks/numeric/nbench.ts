// Tight numeric loops over arrays: the code a vectorizer is for. Six kernels over flat arrays,
// each repeated until it takes a measurable time, each ending in a check value every build prints
// (the Rust and C++ ports in rust/ and cpp/ print the same six).
//
//   saxpy    y[i] += a * x[i]                        element-wise, vectorizable as written
//   triad    z[i] = x[i] * y[i] + c                  element-wise, three arrays
//   stencil  out[i] = (in[i-1] + 2 in[i] + in[i+1]) / 4   neighbours of the same array
//   dot      s += x[i] * y[i]                        a floating-point reduction: strict IEEE
//                                                    order keeps it scalar without --fast-math
//   matmul   c[i][j] += a[i][k] * b[k][j]            64x64, flat arrays, i-k-j order
//   hash     h[i] = (h[i] * 31 + g[i]) & 0xffff      integers, with an integer reduction
//
// No kernel depends on the order of floating-point operations being changed, so every build
// gives the same six numbers as long as it does not contract a*b+c into a fused multiply-add.
declare function floor(v: number): number;
declare function fmod(a: number, b: number): number;

const N = 4096;
const M = 64;

// a value's digits folded into an integer below 999983, which print() shows exactly
function check(v: number): number {
    return fmod(floor(v * 1000.0), 999983.0);
}

function makeArray(n: int, mul: number, mod: int, add: number): number[] {
    let a: number[] = [];
    a.length = n;
    for (let i = 0; i < n; i++) {
        a[i] = (i % mod) * mul + add;
    }

    return a;
}

function sum(a: number[]): number {
    let s = 0.0;
    for (let i = 0; i < a.length; i++) {
        s += a[i];
    }

    return s;
}

function saxpy(a: number, x: number[], y: number[], n: int) {
    for (let i = 0; i < n; i++) {
        y[i] += a * x[i];
    }
}

function triad(x: number[], y: number[], z: number[], c: number, n: int) {
    for (let i = 0; i < n; i++) {
        z[i] = x[i] * y[i] + c;
    }
}

function stencil(src: number[], dst: number[], n: int) {
    for (let i = 1; i < n - 1; i++) {
        dst[i] = (src[i - 1] + 2.0 * src[i] + src[i + 1]) * 0.25;
    }
}

function dot(x: number[], y: number[], n: int): number {
    let s = 0.0;
    for (let i = 0; i < n; i++) {
        s += x[i] * y[i];
    }

    return s;
}

function matmul(a: number[], b: number[], c: number[], m: int) {
    for (let i = 0; i < m; i++) {
        for (let k = 0; k < m; k++) {
            const aik = a[i * m + k];
            for (let j = 0; j < m; j++) {
                c[i * m + j] += aik * b[k * m + j];
            }
        }
    }
}

function hash(h: int[], g: int[], n: int): int {
    let s: int = 0;
    for (let i = 0; i < n; i++) {
        h[i] = (h[i] * 31 + g[i]) & 0xffff;
        s += h[i];
    }

    return s;
}

function runSaxpy(): number {
    const x = makeArray(N, 0.5, 97, 1.0);
    const y = makeArray(N, 0.25, 89, 2.0);
    for (let r = 0; r < 60000; r++) {
        saxpy(0.000001, x, y, N);
    }

    return check(sum(y));
}

function runTriad(): number {
    const x = makeArray(N, 0.5, 97, 1.0);
    const y = makeArray(N, 0.25, 89, 2.0);
    const z = makeArray(N, 0.0, 1, 0.0);
    let s = 0.0;
    for (let r = 0; r < 60000; r++) {
        triad(x, y, z, r * 0.001, N);
        s += z[r % N];
    }

    return check(s + sum(z));
}

function runStencil(): number {
    let a = makeArray(N, 0.5, 97, 1.0);
    let b = makeArray(N, 0.5, 97, 1.0);
    for (let r = 0; r < 30000; r++) {
        stencil(a, b, N);
        stencil(b, a, N);
    }

    return check(sum(a));
}

function runDot(): number {
    const x = makeArray(N, 0.5, 97, 1.0);
    const y = makeArray(N, 0.25, 89, 2.0);
    let s = 0.0;
    for (let r = 0; r < 30000; r++) {
        x[r % N] += 0.5;
        s += dot(x, y, N) * 0.000001;
    }

    return check(s);
}

function runMatmul(): number {
    const a = makeArray(M * M, 0.001, 13, 0.01);
    const b = makeArray(M * M, 0.002, 7, 0.02);
    const c = makeArray(M * M, 0.0, 1, 0.0);
    for (let r = 0; r < 400; r++) {
        matmul(a, b, c, M);
    }

    return check(sum(c));
}

function runHash(): number {
    let h: int[] = [];
    let g: int[] = [];
    h.length = N;
    g.length = N;
    for (let i = 0; i < N; i++) {
        h[i] = i % 251;
        g[i] = i % 241 + 7;
    }

    let s: int = 0;
    for (let r = 0; r < 60000; r++) {
        s = (s + hash(h, g, N)) & 0xfffff;
    }

    return s;
}

// With no argument every kernel runs; with a kernel's name only that one does, so bench.sh can
// time them one by one.
function main(argc: int, args: Ref<string>): int {
    const only = argc > 1 ? Deref(args[1]) : "";
    const all = argc <= 1;
    print("start...");
    if (all || only == "saxpy") print("saxpy:", runSaxpy());
    if (all || only == "triad") print("triad:", runTriad());
    if (all || only == "stencil") print("stencil:", runStencil());
    if (all || only == "dot") print("dot:", runDot());
    if (all || only == "matmul") print("matmul:", runMatmul());
    if (all || only == "hash") print("hash:", runHash());
    print("done.");
    return 0;
}

// Run by test-compile-fast-math / test-jit-fast-math with the extra --fast-math
// compiler flag (see test-runner -fast-math). Verifies that the relaxed
// floating-point mode keeps the language semantics it promises to keep:
//
// 1. FP loop reductions still compute correct results. All values below are
//    exact multiples of 0.125, so every partial sum is exactly representable
//    and reassociation (the transform --fast-math licenses, and what the -O3
//    vectorizer uses to turn the loop into SIMD lanes) cannot change the
//    result - exact equality asserts stay valid.
// 2. NaN and Infinity semantics survive: --fast-math deliberately omits
//    nnan/ninf, so `x !== x` NaN checks and Infinity arithmetic keep working.
// 3. Plain counted integer loops are unaffected.

function isnan(x: number) {
    return x !== x;
}

function fpReduction(n: int): number {
    let arr: number[] = [];
    for (let i = 0; i < n; i++) {
        arr.push((i % 10) * 0.125);
    }

    let sum: number = 0.0;
    for (let i = 0; i < n; i++) {
        sum = sum + arr[i];
    }

    return sum;
}

function dotProduct(n: int): number {
    let a: number[] = [];
    let b: number[] = [];
    for (let i = 0; i < n; i++) {
        a.push((i % 8) * 0.25);
        b.push(((i + 1) % 4) * 0.5);
    }

    let sum: number = 0.0;
    for (let i = 0; i < n; i++) {
        sum = sum + a[i] * b[i];
    }

    return sum;
}

function intTriangle(n: int): int {
    let s: int = 0;
    for (let i: int = 1; i <= n; i++) {
        s = s + i;
    }

    return s;
}

function testNaNAndInfinity() {
    const zero: number = fpReduction(10) - 5.625; // 0.0, computed at runtime
    const nan = zero / zero;
    assert(nan !== nan, "NaN !== NaN must stay true under --fast-math");
    assert(!(nan === nan), "NaN === NaN must stay false under --fast-math");
    assert(isnan(nan), "isnan(0/0)");
    assert(!isnan(1.5), "!isnan(1.5)");
    assert(!isnan(zero), "!isnan(0)");

    const inf = 1.0 / zero;
    assert(!isnan(inf), "Infinity is not NaN");
    assert(inf > 1.0e308, "Infinity compares greater than any finite double");
    assert(1.0 / inf === 0.0, "1/Infinity === 0");
    assert(-inf < -1.0e308, "-Infinity stays representable");
}

function main() {
    // per 10 elements: (0+1+...+9)*0.125 = 5.625; 100 groups
    assert(fpReduction(1000) === 562.5, "fp sum reduction");

    // per 8 elements the products sum to exactly 5.0; 100 cycles
    assert(dotProduct(800) === 500.0, "fp dot product");

    assert(intTriangle(1000) === 500500, "integer counted loop");

    testNaNAndInfinity();

    print("done.");
}

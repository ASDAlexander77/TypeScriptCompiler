// Arithmetic on a `number | undefined` that has no value gives NaN, as JavaScript does (it gave the default,
// 0: `m + 1` was 1); one that has a value gives the value's. A bitwise or shift operator reads undefined as 0,
// as JavaScript does too.

function add(m: number | undefined, k: number) {
    return m + k;
}

function main() {
    let m: number | undefined = undefined;
    assert(isNaN(m + 1), "m + 1");
    assert(isNaN(1 - m), "1 - m");
    assert(isNaN(m * 2) && isNaN(m / 2) && isNaN(m % 2), "* / %");
    assert(isNaN(add(undefined, 3)), "a parameter");
    assert((m | 0) == 0, "bitwise reads 0");

    m = 4;
    assert(m + 1 == 5 && 10 - m == 6 && m * 2 == 8, "with a value");
    assert(add(2, 3) == 5, "a parameter with a value");

    print("done.");
}

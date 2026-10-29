// A bigint is the language's 64-bit integer. Its literal is a `bigint` (it was an i64): widened it is one,
// so `x ?? 7n` of an optional bigint is a bigint (it was `bigint | i64`, and the null test an i64 against a
// pointer), and `5n` given for `bigint | string` is its bigint member (an invalid cast). `/`, `%`, shifts
// and bitwise operators stay 64-bit integers: `10n / 3n` was 3.33333 through a number, and `1n << 40n`
// shifted by 40 mod 32 through an i32. A division by 0n or a negative power stops the program (RangeError).
function orSeven(x?: bigint) {
    return x ?? 7n;
}

function next(u: bigint | string) {
    if (typeof u == "bigint") {
        return u + 1n;
    }

    return 0n;
}

function main() {
    const big = 1234567890123456789n;
    assert(`${big}` == "1234567890123456789", "a literal near the top of the range");

    assert(orSeven() == 7n, "?? of no value");
    assert(orSeven(3n) == 3n, "?? of a value");

    assert(next(5n) == 6n, "a literal given for bigint | string");
    assert(next("s") == 0n, "a string given for bigint | string");

    const a = 10n;
    const b = 3n;
    assert(a / b == 3n, "division truncates");
    assert(-a / b == -3n, "toward zero");
    assert(a % b == 1n && -a % b == -1n, "remainder");
    assert(1n << 40n == 1099511627776n, "a shift past 32 bits");
    assert((1n << 62n) >> 60n == 4n, "a right shift");
    assert((a & 6n) == 2n && (a | 5n) == 15n && (a ^ 3n) == 9n, "bitwise");

    let c = 100n;
    c /= 7n;
    c <<= 33n;
    assert(c == 120259084288n, "compound assignments");

    // `**` is exact: through a number `2n ** 62n` was rounded
    const two = 2n;
    assert(two ** 62n == 4611686018427387904n, "a power past 2^53");
    assert(3n ** 0n == 1n && (-3n) ** 3n == -27n && 10n ** 18n == 1000000000000000000n, "powers");
    let p = 2n;
    p **= 5n;
    assert(p == 32n, "**=");

    print("done.");
}

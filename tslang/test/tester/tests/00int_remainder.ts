// regression test: `%` of two integers of any width is an integer remainder with an integer result.
//
// bug found and fixed: only a 64-bit operand made `%` an integer remainder (00int64_remainder.ts);
// two narrower ones were cast to number, so a hash modulo the length of a table - every lookup of a
// Map - was an fmod call and two conversions, and modulo a length of 0 it was NaN made into an index.

function bucket(hashCode: int, buckets: int[]): int {
    return buckets[<uint>hashCode % <uint>buckets.length];
}

function main() {
    // signed: the remainder takes the dividend's sign
    let i: s32 = 1000;
    assert(i % 7 == 6);
    let minus: s32 = -1000;
    assert(minus % 7 == -6);
    assert(i % -7 == 6);
    assert(minus % -7 == -6);

    // the result is an integer: it indexes, and it stays one in an integer
    const r = i % 7;
    const values: s32[] = [0, 1, 2, 3, 4, 5, 6];
    assert(values[r] == 6);
    let kept: s32 = i % 7;
    assert(kept == 6);

    // unsigned: all 32 bits set is not -1
    let max: u32 = 4294967295;
    assert(max % 10 == 5);
    assert(max % 7 == 3);
    let top: u32 = 2147483648;
    const big: u32 = 3000000000;
    assert(big % top == 852516352);

    // a literal takes the other operand's type, on either side
    let u: u32 = 4000000000;
    assert(u % 1000 == 0);
    assert(u % 7 == 3);
    let seven: u32 = 7;
    assert(4000000000 % seven == 3);

    // a signed and an unsigned one: neither changes its value
    const three: u32 = 3;
    assert(minus % three == -1);
    assert(max % minus == 295);
    assert(u % -7 == 3);

    // what `>>>` gives is unsigned: all 32 bits set, not -1
    let all: s32 = -1;
    assert((all >>> 0) % 5 == 0);
    assert((all >>> 0) % seven == 3);
    assert((all >>> 0) % i == 295);

    // narrower ones
    let b: u8 = 250;
    let c: u8 = 7;
    assert(b % c == 5);
    let small: s8 = -100;
    assert(small % 7 == -2);
    let h: s16 = -30000;
    let k: u16 = 60000;
    assert(h % 7 == -5);
    assert(k % 7 == 3);
    assert(h % k == -30000);

    // of different widths: the wider one
    let wide: s64 = -5000000000;
    assert(wide % i == 0);
    assert(i % wide == 1000);

    // compound
    let acc: s32 = 1000;
    acc %= 7;
    assert(acc == 6);
    let uacc: u32 = 4000000000;
    uacc %= 7;
    assert(uacc == 3);

    // a hash modulo the length of a table
    const buckets: int[] = [10, 11, 12, 13, 14];
    assert(bucket(7, buckets) == 12);
    assert(bucket(-1, buckets) == 10);   // 4294967295 % 5

    // a number on either side is still a number's remainder
    let n: number = 7.5;
    assert(n % 2 == 1.5);
    assert(i % 2.5 == 0);
    assert(7 % n == 7);
    const zero: number = 0;
    assert(isNaN(n % zero));

    print("done.");
}

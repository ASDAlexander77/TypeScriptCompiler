// regression test: `%` of 64-bit integers is an integer remainder.
//
// bug found and fixed: `%` cast both operands to number (f64), so a u64 remainder was wrong once the
// value passed 2^53 (the low byte of "main_vs\0" read 108 instead of 109). An unsigned operand
// also has to take the unsigned remainder.

function main() {
    const BYTE: u64 = 256;
    let w: u64 = 32499774568096109;     // the bytes of "main_vs\0", little-endian
    const m: u64 = 109;
    assert(w % BYTE == m);
    assert(w % 256 == m);

    let r: u64 = 32499774568096109;
    r %= 256;
    assert(r == m);

    // read every byte back
    const expected = [109, 97, 105, 110, 95, 118, 115, 0];
    let word: u64 = 32499774568096109;
    for (const e of expected) {
        const byte: u64 = e;
        assert(word % BYTE == byte);
        word = word >> 8;
    }

    // all 64 bits set: unsigned, not -1
    let max: u64 = 18446744073709551615;
    const five: u64 = 5;
    const one: u64 = 1;
    assert(max % 10 == five);
    assert(max % 7 == one);

    // signed: the remainder takes the dividend's sign
    let s: s64 = -32499774568096109;
    const minus109: s64 = -109;
    const minus340597691: s64 = -340597691;
    assert(s % 256 == minus109);
    assert(s % 1000000007 == minus340597691);

    // narrower integers keep the number path
    let i: s32 = 1000;
    assert(i % 7 == 6);
    let n: number = 7.5;
    assert(n % 2 == 1.5);

    print("done.");
}

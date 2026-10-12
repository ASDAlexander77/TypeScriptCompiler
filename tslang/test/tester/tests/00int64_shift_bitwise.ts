// regression test: shifts and bitwise operators on 64-bit integers keep the 64-bit width.
//
// bug found and fixed: `<<`, `>>`, `>>>`, `&`, `|` and `^` cast both operands to i32 (the
// JavaScript Int32 coercion of `number`) whatever their type, so `u64 << 40` shifted by
// 40 mod 32 = 8 and gave 256, and the upper half of a u64 was lost in `&`/`|`/`^`. Also `>>`
// of an unsigned operand shifted the sign bit in.

function main() {
    let a: u64 = 1;
    const twoTo40: u64 = 1099511627776;
    const twoTo33: u64 = 8589934592;

    const b = a << 40;
    print(b);
    assert(b == twoTo40);

    let c: u64 = 1;
    c = c << 33;
    assert(c == twoTo33);

    let g: u64 = 1;
    g <<= 40;
    assert(g == twoTo40);

    let i: s64 = 1;
    const twoTo40s: s64 = 1099511627776;
    assert(i << 40 == twoTo40s);

    // right shifts of a 64-bit value
    const one: u64 = 1;
    let f: u64 = 1099511627776;
    const r1: u64 = 128;
    assert(f >> 33 == r1);
    assert(f >>> 33 == r1);

    // an unsigned `>>` shifts zeros in
    let h: u64 = 1;
    h = h << 63;
    assert(h >> 63 == one);
    assert(h >>> 63 == one);

    const three: u64 = 3;
    let k: u64 = 3;
    assert(k << 62 >> 62 == three);

    // a signed `>>` keeps the sign, `>>>` doesn't
    let s: s64 = -8;
    const minusFour: s64 = -4;
    const fifteen: s64 = 15;
    assert(s >> 1 == minusFour);
    assert(s >>> 60 == fifteen);

    // bitwise operators keep the upper half
    let m: u64 = 0xFFFFFFFFFF;
    const upper: u64 = 0xFF00000000;
    const zero: u64 = 0;
    assert((m & 0xFF00000000) == upper);
    assert((m | 1) == m);
    assert((m ^ m) == zero);

    // a `number` operand still shifts by its count mod 32, as in JavaScript
    let n = 1;
    assert(n << 33 == 2);
    assert(1 << 33 == 2);
    let x = -8;
    assert(x >> 1 == -4);
    assert(x >>> 28 == 15);

    print("done.");
}

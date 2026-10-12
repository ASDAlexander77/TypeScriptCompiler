// regression test: a union of literals, as a ternary of two literals is, takes part in arithmetic
// as its literals' type.
//
// bug found and fixed: `c ? 0.5 : 1.0` is the union `0.5 | 1`, and a binary operator cast that
// union to the other operand's type: `int * (c ? 0.5 : 1.0)` multiplied two s32s, 0.5 became 0,
// and the product was 0.

function scale(split: boolean, width: int): number {
    return width * (split ? 0.5 : 1.0);
}

function main() {
    let split = true;
    let held = false;
    const width = 100;

    assert(width * (split ? 0.5 : 1.0) == 50);
    assert((split ? 0.5 : 0.25) * width == 50);
    assert(width * (held ? 0.5 : 0.25) == 25);
    assert(100 * (split ? 0.5 : 1.0) == 50);
    assert(width + (split ? 0.5 : 1.0) == 100.5);
    assert(width - (split ? 0.5 : 1.0) == 99.5);
    assert(scale(true, 100) == 50);
    assert(scale(false, 100) == 100);

    const w: number = 100;
    assert(w * (split ? 0.5 : 1.0) == 50);

    // through a variable
    const half = split ? 0.5 : 1.0;
    assert(width * half == 50);

    const left = split ? 448.0 : 96.5;
    assert(left + 10.5 == 458.5);
    let m = left * 2;
    m = m / 3;
    assert(m > 298.6 && m < 298.7);

    assert((held ? 1.0 : 0.0) - (split ? 1.0 : 0.0) == -1);

    // comparisons and strings still work
    assert(half == 0.5);
    assert(half != 1.0);
    const s = held ? "a" : "b";
    assert(s + "!" == "b!");
    assert(s == "b");

    print("done.");
}

// Two `any`s holding numbers compare by value, as JavaScript numbers do, whatever types they were
// boxed with: a `number` and an integer were neither `===` nor `!==` each other, so a check like
// `isEq(x: any, y: any) { if (x !== y) fail(); }` could never fail on a computed double compared
// with an integer literal.

function isNe(x: any, y: any): boolean {
    return x !== y;
}

function isStrictEq(x: any, y: any): boolean {
    return x === y;
}

function mixedWidths() {
    const n: any = 2.5;
    const i: any = 1;
    assert(n !== i, "2.5 !== 1");
    assert(!(n === i), "2.5 === 1");
    assert(n != i, "2.5 != 1");
    assert(!(n == i), "2.5 == 1");

    const whole: any = 1.0;
    assert(whole === i, "1.0 === 1");
    assert(!(whole !== i), "1.0 !== 1");
    assert(whole == i, "1.0 == 1");

    assert(isNe(2.2e-308, -1), "isNe(2.2e-308, -1)");
    assert(!isNe(-1.0, -1), "isNe(-1.0, -1)");
    assert(isStrictEq(3, 3.0), "isStrictEq(3, 3.0)");
}

function widths() {
    const a: any = <s64>5;
    const b: any = <u8>5;
    const c: any = <f32>5;
    const d: any = 5.0;
    const e: any = <s16>-3;
    const f: any = -3.0;
    assert(a === b, "s64 === u8");
    assert(b === c, "u8 === f32");
    assert(c === d, "f32 === number");
    assert(e === f, "s16 === number (negative)");
    assert(a !== e, "s64 !== s16");
}

function ordering() {
    const big: any = 2.5;
    const small: any = 1;
    const neg: any = -7;
    assert(big > small, "2.5 > 1");
    assert(small < big, "1 < 2.5");
    assert(neg < small, "-7 < 1");
    assert(big >= 2.5, "2.5 >= 2.5");
    assert(!(neg > small), "-7 > 1");
}

function nan() {
    const x: any = NaN;
    const y: any = NaN;
    assert(x !== y, "NaN !== NaN");
    assert(!(x === y), "NaN === NaN");
}

function otherKinds() {
    // strings, and values of different kinds, are unaffected
    const s: any = "a";
    const t: any = "a";
    const n: any = 1;
    assert(s === t, "\"a\" === \"a\"");
    assert(s !== n, "\"a\" !== 1");
    assert(!(s === n), "\"a\" === 1");
    const b: any = true;
    assert(b !== n, "true !== 1");
    assert(b == n, "true == 1");
}

function main() {
    mixedWidths();
    widths();
    ordering();
    nan();
    otherKinds();
    print("done.");
}

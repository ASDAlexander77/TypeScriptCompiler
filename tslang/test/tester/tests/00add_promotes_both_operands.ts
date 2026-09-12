// regression test: `+` must promote both operands to the wider type, like every other
// arithmetic operator, instead of coercing the right operand to the left operand's type.
//
// `+` is the one arithmetic operator that is also string concatenation, so it chose its
// result type from the left operand before looking at the right one, and then cast the
// right one to match. Where the right operand was the WIDER of the two, that cast
// truncated it -- and the truncation happened before the addition rather than after it,
// so the fraction (or the high bits) were gone by the time the two were added:
//
//   2 + 3.5        read 5     (3.5 truncated to 3, then 2 + 3)
//   1 + (-0.5)     read 1     (-0.5 truncated to 0; truncating the SUM would give 0)
//   true + 2.5     read 2     (2.5 dragged down to a boolean)
//   i8(100) + i32(1000) read 76  (1100 wrapped into i8)
//
// `-`, `*` and `/` were all correct, because they share a promotion path that picks the
// wider of the two operand types. `+` now uses that same path whenever both operands are
// unambiguously numeric; every other shape (strings, `any`, unions, `[Symbol.toPrimitive]`
// objects) keeps the left-preferring behaviour that string concatenation depends on.

function main() {
    // the reported shape: integer on the left, float on the right
    let i = 2;
    let f = 3.5;
    assert(i + f == 5.5, "i + f");
    assert(f + i == 5.5, "f + i");

    // constant operands are not spared -- this was wrong at compile time too
    assert(2 + 3.5 == 5.5, "literals");

    // the fraction is discarded before the addition, not after: truncating the sum of
    // 1 + (-0.5) would give 0, so a result of exactly 0.5 is what distinguishes them
    assert(1 + (-0.5) == 0.5, "negative fraction");

    // the other arithmetic operators, which were already right and must stay right
    assert(i * f == 7.0, "i * f");
    assert(f - i == 1.5, "f - i");
    assert(i / f > 0.571 && i / f < 0.5715, "i / f");

    // booleans widen to number first, so a boolean on the left cannot drag a float down
    assert(true + 2.5 == 3.5, "true + float");
    assert(2.5 + true == 3.5, "float + true");
    assert(true + true == 2, "true + true");
    assert(true + 1 == 2, "true + int");
    assert(2 + true == 3, "int + true");

    // narrowing between integer widths is the same bug without any float involved
    let narrow: i8 = 100;
    let wide: i32 = 1000;
    assert(narrow + wide == 1100, "i8 + i32");
    assert(wide + narrow == 1100, "i32 + i8");

    // integer + integer stays exact
    assert(2 + 3 == 5, "int literals");
    let q = 7;
    let r = 2;
    assert(q + r == 9, "int vars");

    // string concatenation is what the left-preferring rule exists for, and is unchanged
    assert("fo" + 1 == "fo1", "string + int");
    assert(1 + "fo" == "1fo", "int + string");
    assert("n=" + 3.5 == "n=3.5", "string + float");
    let s = "x";
    assert(s + 2 == "x2", "string var + int");

    print("done.");
}

// parseInt, parseFloat and a cast from a string to a number parse with strtoll/strtod: atoi and
// atof leave a value out of range undefined (MSVC saturated, glibc wrapped), and strtol returns a
// `long`, 64 bits on Linux, which was read as 32. A 32-bit result saturates at the i32 range.
function main() {
    assert(parseInt("42") == 42, "parseInt");
    assert(parseInt("  -17abc") == -17, "leading spaces, sign, trailing text");
    assert(parseInt("abc") == 0, "no digits");
    assert(parseInt("010") == 10, "decimal, not octal, without a radix");
    assert(parseInt("ff", 16) == 255, "radix 16");
    assert(parseInt("-101", 2) == -5, "radix 2");

    assert(parseInt("2147483647") == 2147483647, "i32 max");
    assert(parseInt("-2147483648") == -2147483648, "i32 min");
    assert(parseInt("3000000000") == 2147483647, "above i32 saturates");
    assert(parseInt("-3000000000") == -2147483648, "below i32 saturates");
    assert(parseInt("99999999999999999999999") == 2147483647, "above i64 saturates");
    assert(parseInt("100000000", 16) == 2147483647, "radix 16 above i32 saturates");

    const s = "123";
    assert(<i32>s == 123, "cast to i32");
    assert(<i64>"9007199254740993" == 9007199254740993, "cast to i64");

    assert(parseFloat("2.5e3") == 2500, "parseFloat");
    assert(parseFloat("  -0.25x") == -0.25, "leading spaces, trailing text");
    assert(parseFloat("1e400") > 1e308, "above the double range is infinite");
    assert(parseFloat("1e-400") == 0, "below the double range is 0");
    assert(<f32>"2.5" == 2.5, "cast to f32");

    print("done.");
}

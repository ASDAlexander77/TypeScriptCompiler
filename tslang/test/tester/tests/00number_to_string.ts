// A number becomes text as JS writes it (Number::toString): the fewest digits that read back as
// the same double, plain decimal while the decimal exponent n is in -6 < n <= 21, `d.ddde+N` /
// `d.ddde-N` otherwise, NaN / Infinity / -Infinity, and "0" for -0. `%g` kept 6 digits (#780).
// (String(x) is the default library's, over the same cast; these tests build without it.)
function concat(x: number) {
    return "" + x;
}

function template(x: number) {
    return `${x}`;
}

function converted(x: number) {
    return <string>x;
}

function check(x: number, expected: string) {
    const viaConcat = concat(x);
    const viaTemplate = template(x);
    const viaCast = converted(x);
    if (viaConcat != expected || viaTemplate != expected || viaCast != expected) {
        print("expected", expected, "got", viaConcat, viaTemplate, viaCast);
        assert(false, "number to string");
    }
}

function main() {
    const two: number = 2;
    const zero: number = 0;
    const one: number = 1;
    const three: number = 3;

    check(0.1 + 0.2, "0.30000000000000004");
    check(one / three, "0.3333333333333333");
    check(two / three, "0.6666666666666666");
    check(123456789.123456789, "123456789.12345679");
    check(two ** 53 + 2, "9007199254740994");
    check(two ** 53, "9007199254740992");
    check(two ** 60, "1152921504606847000");
    check(1e-7, "1e-7");
    check(1.5e-7, "1.5e-7");
    check(0.000001, "0.000001");
    check(0.0000012345, "0.0000012345");
    check(123e-20, "1.23e-18");
    check(5e-324, "5e-324");
    check(-5e-324, "-5e-324");
    check(2.2250738585072014e-308, "2.2250738585072014e-308");
    check(1.7976931348623157e308, "1.7976931348623157e+308");
    check(1e21, "1e+21");
    check(1e20, "100000000000000000000");
    check(1.2345678901234568e20, "123456789012345680000");
    check(1e23, "1e+23");
    check(8.41e21, "8.41e+21");
    check(4.35, "4.35");
    check(0.1, "0.1");
    check(-0.1, "-0.1");
    check(1.5, "1.5");
    check(-2.5, "-2.5");
    check(100, "100");
    check(-100, "-100");
    check(1, "1");
    check(123.456, "123.456");
    check(0.5, "0.5");
    check(1e100, "1e+100");
    check(-1.5e-10, "-1.5e-10");
    check(zero, "0");
    check(-zero, "0");
    check(zero / zero, "NaN");
    check(one / zero, "Infinity");
    check(-one / zero, "-Infinity");

    // a literal, not only a value computed at run time
    assert(`${0.1}` == "0.1", "literal in a template");
    assert("" + 1e21 == "1e+21", "literal concatenated");
    assert(<string>1e-7 == "1e-7", "literal cast to string");

    print(0.1 + 0.2);
    print(one / three);
    print(1e21, 1e-7, -zero);
    print(zero / zero, one / zero, -one / zero);

    print("done.");
}

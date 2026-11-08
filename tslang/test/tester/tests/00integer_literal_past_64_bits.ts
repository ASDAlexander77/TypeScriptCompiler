// An integer literal past 64 bits is a `number`, the nearest double, as in JS: it was cut to its
// low 64 bits (123456789012345680000 read -5670419503621181000), and past 128 bits it was an error.
function main() {
    const big = 123456789012345680000;
    assert(`${big}` == "123456789012345680000", "past 64 bits");
    assert(big > 0, "past 64 bits: positive");
    assert(big / 1e20 == 1.234567890123457, "past 64 bits: arithmetic as a number");

    // up to 2^64 - 1 a literal is still a u64, exactly
    const twoTo63 = 9223372036854775808;
    assert(`${twoTo63}` == "9223372036854775808", "2^63: u64");
    const maxU64 = 18446744073709551615;
    assert(`${maxU64}` == "18446744073709551615", "2^64 - 1: u64");

    const twoTo64 = 18446744073709551616;
    assert(`${twoTo64}` == "18446744073709552000", "2^64");

    const negative = -123456789012345680000;
    assert(`${negative}` == "-123456789012345680000", "negative past 64 bits");

    const past128 = 1000000000000000000000000000000000000000000;
    assert(`${past128}` == "1e+42", "past 128 bits");

    const pastDouble = 100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000;
    assert(`${pastDouble}` == "Infinity", "past the largest double");

    // still an integer below 2^63
    const fits = 9007199254740993;
    assert(`${fits}` == "9007199254740993", "fits s64: stays an integer");

    print(big, twoTo63);
    print("done.");
}

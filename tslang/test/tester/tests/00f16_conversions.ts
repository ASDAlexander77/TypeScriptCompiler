// f16 to and from number and f32 in an executable: LLVM calls compiler-rt's __truncdfhf2, __truncsfhf2
// and __extendhfsf2 for these, and MSVC's runtime has none of them, so the link failed on Windows
function toHalf(x: number): f16 {
    return <f16>x;
}

function toHalf32(x: f32): f16 {
    return <f16>x;
}

function back(h: f16): number {
    return <number>h;
}

function back32(h: f16): f32 {
    return <f32>h;
}

function main() {
    assert(back(toHalf(1.5)) == 1.5, "1.5");
    assert(back(toHalf(-2)) == -2, "-2");
    assert(back(toHalf(65504)) == 65504, "the largest half");
    assert(back(toHalf(65520)) == Infinity, "past the largest rounds to infinity");
    assert(back(toHalf(0.1)) == 0.0999755859375, "0.1, to the nearest half");
    assert(back(toHalf(1 + 1 / 2048)) == 1, "a tie goes to even");
    assert(back(toHalf(5.960464477539063e-8)) == 5.960464477539063e-8, "the smallest subnormal");
    assert(back(toHalf(1e-9)) == 0, "too small is 0");
    assert(1 / back(toHalf(-0)) == -Infinity, "-0 stays -0");
    assert(isNaN(back(toHalf(NaN))), "NaN");
    assert(back32(toHalf32(<f32>3.25)) == <f32>3.25, "through f32");
    print("done.");
}

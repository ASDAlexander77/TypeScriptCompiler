// -x of a float 0 is -0, as in JavaScript, and so is the literal -0: the negation was 0.0 - x, which is
// +0 for 0, and -0 was the integer 0, so 1 / -0 and 1 / -v were Infinity
function negate(v: number): number {
    return -v;
}

function main() {
    assert(1 / negate(0) == -Infinity, "-v of 0 is -0");
    assert(1 / negate(-0) == Infinity, "-v of -0 is 0");
    assert(negate(1.5) == -1.5, "-v of 1.5");
    assert(1 / -0 == -Infinity, "the literal -0");
    const z = -0;
    assert(1 / z == -Infinity, "a constant of -0");
    const f: f32 = 0;
    assert(1 / -f == -Infinity, "an f32");
    let i: int = 0;
    assert(-i == 0, "an int stays an int");
    print("done.");
}

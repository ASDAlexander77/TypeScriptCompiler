// +, - and * of a bigint and a number literal or an int are done in 64 bits: bigint was not on the
// promotion ladder, so the operands met at the literal's s32 and the product was cut to 32 bits
const K = 1000000000;

function main() {
    const s: bigint = 1710505800;
    assert(s * 1000000000 == 1710505800000000000, "bigint * literal");
    assert(1000000000 * s == 1710505800000000000, "literal * bigint");
    assert(s * K == 1710505800000000000, "bigint * const");
    const k: int = 1000000000;
    assert(s * k == 1710505800000000000, "bigint * int");
    const big: bigint = 9000000000000000000;
    assert(big + 1 == 9000000000000000001, "bigint + literal");
    assert(1 + big == 9000000000000000001, "literal + bigint");
    assert(big - 1 == 8999999999999999999, "bigint - literal");
    assert(big > 1, "a comparison keeps the bigint");

    // an int becomes a bigint sign-extended: zero-extended, -1000000 was 4293967296
    const neg: bigint = -1000000;
    assert(neg < 0, "a negative bigint is below 0");
    assert(neg * 2 == -2000000, "a negative bigint * literal");
    const m: int = -5;
    const fromInt: bigint = m;
    assert(fromInt == -5 && fromInt < 0, "a negative int as a bigint");
    assert(s * -1 == -1710505800, "bigint * a negative literal");
    print("done.");
}

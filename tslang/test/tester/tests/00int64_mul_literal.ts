// regression test: arithmetic of a 64-bit unsigned integer and an integer literal is done in 64 bits.
//
// bug found and fixed: the numeric promotion ladder had no unsigned types, so `u64 * 2` found only
// the literal's s32 and multiplied in 32 bits: `2^32 * 2` was 0, and `d *= 256` wrapped to 0 after
// four bytes.

function main() {
    const twoTo32: u64 = 4294967296;
    const twoTo33: u64 = 8589934592;
    const twoTo32Plus2: u64 = 4294967298;
    const twoTo32Minus2: u64 = 4294967294;

    let a: u64 = 4294967296;
    assert(a * 2 == twoTo33);
    assert(2 * a == twoTo33);
    assert(a + 2 == twoTo32Plus2);
    assert(a - 2 == twoTo32Minus2);

    let d: u64 = 16777216;
    d *= 256;
    assert(d == twoTo32);

    // the shape the bug was found in: a std::string's word built a byte at a time
    let word: u64 = 0;
    let shift: u64 = 1;
    const bytes = [109, 97, 105, 110, 95, 118, 115];
    for (const b of bytes) {
        const byte: u64 = b;
        word += byte * shift;
        shift *= 256;
    }

    const mainVs: u64 = 32499774568096109;
    assert(word == mainVs);

    const k: u64 = 256;
    let e: u64 = 16777216;
    assert(e * k == twoTo32);

    print("done.");
}

// Throwing and catching a u32.
//
// A thrown u32 described itself as a signed `int` (`.H` on Windows, `_ZTIi` on Linux), so every
// reader took its top bit for a sign: `catch (e: number)` and an untyped catch both saw
// 0xFFFFFFFF as -1. On Windows the widening and boxing thunks were also shared by name with an
// `int` throw, so a module throwing both used whichever was built first. It is now thrown as a
// C++ `unsigned int` (`.I` / `_ZTIj`), with records and thunks of its own.
//
// This module throws both, on purpose. The values are checked by sign, which cannot mistake -1
// for 0xFFFFFFFF - an equality test against 4294967295 passed on the broken build.

const big: u32 = 0xFFFFFFFF;

function typedU32() {
    let r: u32 = 0;
    try {
        throw big;
    } catch (e: u32) {
        r = e;
    }

    return r;
}

function untypedU32() {
    let r: number = 0;
    try {
        throw big;
    } catch (e) {
        r = <u32>e;
    }

    return r;
}

function u32AsNumber() {
    let r: number = 0;
    try {
        throw big;
    } catch (e: number) {
        r = e;
    }

    return r;
}

function s32AsNumber() {
    let r: number = 0;
    try {
        throw -1;
    } catch (e: number) {
        r = e;
    }

    return r;
}

function untypedS32() {
    let r: s32 = 0;
    try {
        throw -1;
    } catch (e) {
        r = <s32>e;
    }

    return r;
}

function s32PassesU32Catch() {
    let r: number = 0;
    try {
        try {
            throw -2;
        } catch (e: u32) {
            r = 1;
        }
    } catch (e2: number) {
        r = e2;
    }

    return r;
}

function main() {
    // the conversion every reader below goes through: sitofp made it -1
    const widened: number = big;
    assert(widened > 0, "u32 to number is unsigned");

    assert(typedU32() == big, "catch (e: u32) catches a thrown u32");
    assert(untypedU32() > 0, "an untyped catch reads a thrown u32 unsigned");
    assert(u32AsNumber() > 0, "catch (e: number) widens a thrown u32 unsigned");
    assert(s32AsNumber() < 0, "catch (e: number) still widens a thrown s32 signed");
    assert(untypedS32() < 0, "an untyped catch still reads a thrown s32 signed");
    assert(s32PassesU32Catch() == -2, "a thrown s32 passes a u32 catch");

    print("done.");
}

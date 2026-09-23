// Throwing and catching a boolean or a bigint.
//
// `throw true` and `throw 5n` were rejected at compile time ("Not supported type in throw"): the
// RTTI helpers knew no exception type for either. A boolean is thrown as a C++ `bool` (`_N` /
// `_ZTIb`) and a bigint, an i64, as an `__int64` (`_J` / `_ZTIx`), each with the same boxing entry
// the other primitives have for an untyped catch.
//
// Getting there also fixed four bigint gaps, each reachable without exceptions:
//  - boxing a bigint into an `any` crashed the compiler - typeof had no name for it, so the box
//    got no type tag;
//  - unboxing one failed to compile - `<bigint>` of an `any` holding a string parses it, and the
//    parse was an `atoi` (32 bits) where an i64 was needed;
//  - converting one to a string was a bare inttoptr, so `print(x)` of a bigint crashed;
//  - comparing two bigint values was constant false (`u == v` with equal values included).
// And one unrelated to bigint: `<boolean>` of an `any` failed to compile, because the f32
// branch of the generated unbox compared an f32 with an f64 zero.

function typedBool() {
    let r = false;
    try {
        throw true;
    } catch (e: boolean) {
        r = e;
    }

    return r;
}

function untypedBool() {
    let r = true;
    try {
        throw false;
    } catch (e) {
        r = <boolean>e;
    }

    return r;
}

function boolVariable() {
    let b = 1 < 2;
    let r = false;
    try {
        throw b;
    } catch (e: boolean) {
        r = e;
    }

    return r;
}

function typedBigInt() {
    let r: bigint = 0n;
    try {
        throw 123456789012n;
    } catch (e: bigint) {
        r = e;
    }

    return r;
}

function untypedBigInt() {
    let r: bigint = 0n;
    try {
        throw -5n;
    } catch (e) {
        r = <bigint>e;
    }

    return r;
}

function mismatchPassesOn() {
    let path = 0;
    try {
        try {
            throw 7n;
        } catch (e: boolean) {
            path += 1;
        }
    } catch (e2: bigint) {
        path += e2 == 7n ? 10 : 100;
    }

    try {
        try {
            throw true;
        } catch (e3: bigint) {
            path += 1000;
        }
    } catch (e4: boolean) {
        path += e4 ? 20 : 200;
    }

    return path;
}

function bigIntWithoutExceptions() {
    let x: bigint = -123456789012n;
    let a: any = x;
    let s: string = "v=" + x;
    let fromString: any = "42";
    return typeof a == "bigint" && <bigint>a == x && s == "v=-123456789012" && <bigint>fromString == 42n;
}

function bigIntComparisons() {
    let u: bigint = -123456789012n;
    let v: bigint = -123456789012n;
    let w: bigint = 7n;
    return u == v && u === v && !(u != v) && u < w && w > u && u <= v && w >= u && !(w < u);
}

function main() {
    assert(typedBool(), "catch (e: boolean) catches a thrown boolean");
    assert(!untypedBool(), "an untyped catch reads a thrown boolean");
    assert(boolVariable(), "a boolean variable can be thrown");
    assert(typedBigInt() == 123456789012n, "catch (e: bigint) catches a thrown bigint");
    assert(untypedBigInt() == -5n, "an untyped catch reads a thrown bigint");
    assert(mismatchPassesOn() == 30, "boolean and bigint catches let each other's exceptions through");
    assert(bigIntWithoutExceptions(), "a bigint boxes, unboxes and converts to a string");
    assert(bigIntComparisons(), "bigint values compare as signed integers");

    print("done.");
}

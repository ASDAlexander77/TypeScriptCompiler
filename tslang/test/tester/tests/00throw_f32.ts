// Throwing and catching an f32.
//
// A thrown f32 described itself as a double: on Windows it used the `.N` records with a size of
// 8, on Linux the `_ZTId` type_info. So `catch (e: number)` took a thrown f32 and read eight
// bytes of a four-byte value - 1.5f came back as 0.125. It is now thrown as a C++ `float` (`.M`
// / `_ZTIf`), and a `number` catch takes it by widening, the same way it takes a thrown int.

function typedF32() {
    let f: f32 = 1.5;
    let r: f32 = 0;
    try {
        throw f;
    } catch (e: f32) {
        r = e;
    }

    return r;
}

function untypedF32() {
    let f: f32 = 2.5;
    let r: f32 = 0;
    try {
        throw f;
    } catch (e) {
        r = <f32>e;
    }

    return r;
}

function f32AsNumber() {
    let f: f32 = 1.5;
    let r: number = 0;
    try {
        throw f;
    } catch (e: number) {
        r = e;
    }

    return r;
}

function numberPassesF32Catch() {
    let r: number = 0;
    try {
        try {
            throw 2.25;
        } catch (e: f32) {
            r = -1;
        }
    } catch (e2: number) {
        r = e2;
    }

    return r;
}

function main() {
    assert(typedF32() == 1.5, "catch (e: f32) catches a thrown f32");
    assert(untypedF32() == 2.5, "an untyped catch reads a thrown f32");
    assert(f32AsNumber() == 1.5, "catch (e: number) widens a thrown f32");
    assert(numberPassesF32Catch() == 2.25, "a thrown number passes an f32 catch");

    print("done.");
}

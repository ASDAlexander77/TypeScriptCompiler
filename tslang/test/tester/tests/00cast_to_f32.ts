// Casts to f32 that failed to compile, all reached by unboxing an `any` as an f32 (`<f32>e` in
// an untyped catch): `___unbox<f32>` tries every boxed type, and three of the casts were bad.
// - boolean -> f32 had no path (only boolean -> number did) and became undef with a warning;
// - index -> f32 used index.casts, which only make integers, and lowered to an invalid trunc;
// - string -> f32 returned atof's double as the float.

function fromBoolean(b: boolean) {
    return <f32>b;
}

function fromIndex(i: index) {
    return <f32>i;
}

function fromString(s: string) {
    return <f32>s;
}

function fromAny(a: any) {
    return <f32>a;
}

function main() {
    assert(fromBoolean(true) == 1, "<f32>true is 1");
    assert(fromBoolean(false) == 0, "<f32>false is 0");
    assert(fromIndex(7) == 7, "<f32> of an index");
    assert(fromString("2.5") == 2.5, "<f32> of a string");

    let f: f32 = 2.5;
    assert(fromAny(f) == 2.5, "<f32> of an any holding an f32");

    let n: number = 1.5;
    assert(fromAny(n) == 1.5, "<f32> of an any holding a number");

    print("done.");
}

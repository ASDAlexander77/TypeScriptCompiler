// `a ?? b` takes `b` only when `a` is null or undefined - not when it is 0 or "" - and its type is `a`'s
// without them; a number compared with null or undefined is not equal to it (null used to read as 0)
function orMinusOne(n?: number) {
    return n ?? -1;
}

function orDefault(s?: string) {
    return s ?? "default";
}

function isNull(n: number) {
    return n === null;
}

function isNullOptional(n?: number) {
    return n === null;
}

function isNullishOptional(n?: number) {
    return n == null;
}

function isNotNullishOptional(n?: number) {
    return n != null;
}

function main() {
    assert(orMinusOne() == -1, "no value");
    assert(orMinusOne(3) == 3, "value");
    assert(orMinusOne(0) == 0, "zero is a value");

    const n = orMinusOne(0);
    assert(typeof n == "number", "the result is a number");

    assert(orDefault() == "default", "no string");
    assert(orDefault("a") == "a", "string");
    assert(orDefault("") == "", "empty string is a value");

    let y: number | null = null;
    assert((y ?? 8) == 8, "null");
    y = 0;
    assert((y ?? 9) == 0, "zero of number | null");

    let z: string | null | undefined = undefined;
    assert((z ?? "u") == "u", "undefined of a union");
    z = null;
    assert((z ?? "n") == "n", "null of a union");
    z = "";
    assert((z ?? "e") == "", "empty string of a union");

    const five = 5;
    assert((five ?? 3) == 5, "a number is never nullish");

    let opt: number | undefined;
    opt ??= 0;
    assert(opt == 0, "??= assigns");
    opt ??= 7;
    assert(opt == 0, "??= keeps zero");

    let a: any = null;
    assert((a ?? 4) == 4, "any null");

    assert(!isNull(0), "0 === null");
    assert(!isNullOptional(0), "optional 0 === null");
    assert(!isNullOptional(), "optional undefined === null");
    assert(isNullishOptional(), "optional undefined == null");
    assert(!isNullishOptional(0), "optional 0 == null");
    assert(isNotNullishOptional(0), "optional 0 != null");
    assert(!isNotNullishOptional(), "optional undefined != null");

    print("done.");
}

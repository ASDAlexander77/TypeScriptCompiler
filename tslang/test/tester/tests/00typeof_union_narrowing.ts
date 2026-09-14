// `typeof u === "function"` (also "class", "array") names a kind, not a type. For a union the
// members of that kind are known, so `u` is narrowed to them. It was narrowed to Opaque instead:
// a function could not be called and a class field could not be resolved.
class Box {
    value = 5;
}

function one(): number {
    return 1;
}

function main() {
    let f: (() => number) | string = one;
    if (typeof f === "function") {
        assert(f() == 1, "function member is callable");
    } else {
        assert(false, "function: wrong branch");
    }

    f = "abc";
    if (typeof f !== "function") {
        assert(f.length == 3, "string member after !==");
    } else {
        assert(false, "string: wrong branch");
    }

    let c: Box | number = new Box();
    if (typeof c === "class") {
        assert(c.value == 5, "class member field");
    } else {
        assert(false, "class: wrong branch");
    }

    const numbers: number[] = [1, 2, 3];
    let a: number[] | string = numbers;
    if (typeof a === "array") {
        assert(a.length == 3, "array member length");
        assert(a[2] == 3, "array member element");
    } else {
        assert(false, "array: wrong branch");
    }

    print("done.");
}

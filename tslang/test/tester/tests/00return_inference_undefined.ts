// An inferred return type is the union of what the function returns: `undefined` or `null` beside a value
// gives `T | undefined` / `T | null` in either order, and a bare `return;` beside value returns is undefined
function laterUndefined(c: boolean) {
    if (c) return "a";
    return undefined;
}

function firstUndefined(c: boolean) {
    if (c) return undefined;
    return "a";
}

function bareReturn(c: boolean) {
    if (c) return "a";
    return;
}

function integer(c: boolean) {
    if (c) return 1;
    return undefined;
}

function object(c: boolean) {
    if (c) return { v: "x" };
    return undefined;
}

function orNull(c: boolean) {
    if (c) return "a";
    return null;
}

function onlyBare(c: boolean) {
    if (c) return;
    sideEffect = 1;
}

let sideEffect = 0;

function declared(c: boolean): string | undefined {
    if (c) return;
    return "x";
}

class WithBareReturn {
    v = 1;
    constructor(skip: boolean) {
        if (skip) return;
        this.v = 2;
    }
}

function main() {
    assert(laterUndefined(false) === undefined, "later undefined");
    assert(laterUndefined(true) == "a", "later undefined: value");
    assert(firstUndefined(true) === undefined, "first undefined");
    assert(firstUndefined(false) == "a", "first undefined: value");
    assert(bareReturn(false) === undefined, "bare return");
    assert(bareReturn(true) == "a", "bare return: value");
    assert(integer(false) === undefined, "integer");
    assert(integer(true) == 1, "integer: value");
    assert(object(false) === undefined, "object");
    assert(object(true)?.v == "x", "object: value");
    assert(orNull(false) === null, "null");
    assert(orNull(true) == "a", "null: value");

    onlyBare(false);
    assert(sideEffect == 1, "void function with a bare return");

    assert(declared(true) === undefined, "declared, bare return");
    assert(declared(false) == "x", "declared, value");

    assert(new WithBareReturn(true).v == 1 && new WithBareReturn(false).v == 2, "constructor bare return");

    print("done.");
}

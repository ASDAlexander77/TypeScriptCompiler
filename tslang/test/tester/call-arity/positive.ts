function opt(a: number, b: number, c?: number) {
    return c === undefined ? a + b : a + b + c;
}

function def(a: number, b = 5) {
    return a + b;
}

function va(a: number, ...rest: number[]) {
    return a + rest.length;
}

// tslang rejects `void` as a parameter type, so an `undefined` parameter stands in for the
// other type that TypeScript lets a call omit.
function withUndefined(a: number, u: undefined) {
    return a;
}

function main() {
    assert(opt(1, 2) == 3);
    assert(def(1) == 6);
    assert(va(1) == 1);
    assert(va(1, 2, 3) == 3);
    assert(withUndefined(7) == 7);
    print("done.");
}

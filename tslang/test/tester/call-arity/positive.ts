function opt(a: number, b: number, c?: number) {
    return c === undefined ? a + b : a + b + c;
}

function def(a: number, b = 5) {
    return a + b;
}

function va(a: number, ...rest: number[]) {
    return a + rest.length;
}

// tslang additionally treats an `undefined` parameter as omittable; tsc does not (a call
// omitting `b: undefined` is TS2554). tsc's own omittable case is a `void` parameter, but
// that cannot be written in tslang: it rejects `void` as a parameter type.
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

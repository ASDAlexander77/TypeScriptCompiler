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

function fixed(a: number, b: number) {
    return a + b;
}

function main() {
    // a tuple spread fills exactly the parameters it has elements for
    const pair: [number, number] = [1, 2];
    assert(fixed(...pair) == 3);
    assert(opt(1, 2) == 3);
    assert(def(1) == 6);
    assert(va(1) == 1);
    assert(va(1, 2, 3) == 3);
    assert(withUndefined(7) == 7);
    print("done.");
}

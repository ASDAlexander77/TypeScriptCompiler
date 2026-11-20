// A type parameter inferred from `any` and another type is `any`, as TypeScript has it:
// `[].concat([1])` infers concat's T from `any[]` and `s32[][]`, and `any | s32` - a union with
// `any` in it - crashed the compiler when an element was read.
function pick<T>(a: T, b: T) {
    return b;
}

function main() {
    const x = [].concat([1]);
    assert(x.length == 1, "length");
    const e: number = x[0];
    assert(e == 1, "element");

    const y = [].concat([1, 2], [3]);
    assert(y.length == 3, "two arrays");

    const z = [].concat(4);
    assert(z.length == 1, "one value");

    const v: any = 5;
    const w = pick(v, 6);
    assert(w == 6, "any and a number");

    print("done.");
}

// A type parameter inferred from `any` and another type is `any`, as TypeScript has it:
// `[].concat([1])` infers concat's T from `any[]` and `s32[][]`, and `any | s32` - a union with
// `any` in it - crashed the compiler when an element was read. `cat` has the default library's
// `concat` signature, which the tester has no default library for.
function cat<T>(first: T[], ...other: T[][]) {
    let count = first.length;
    for (const item of other) count += item.length;
    let result: T[] = [];
    for (const v of first) result.push(v);
    for (const item of other) {
        for (const v of item) result.push(v);
    }

    return result;
}

function pick<T>(a: T, b: T) {
    return b;
}

function main() {
    const x = cat([], [1]);
    assert(x.length == 1, "length");
    const e: number = x[0];
    assert(e == 1, "element");

    const y = cat([], [1, 2], [3]);
    assert(y.length == 3, "two arrays");

    // an object literal in an array literal given as `any[]` is one element
    const z = cat([], [{ a: 1 }], [{ a: 2 }]);
    assert(z.length == 2, "object literals");

    const v: any = 5;
    const w = pick(v, 6);
    assert(w == 6, "any and a number");

    print("done.");
}

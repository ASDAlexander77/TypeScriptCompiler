// A new array given to an optional parameter (`new Set([a, b])` of DefaultLib's `constructor(values?: V[])`)
// reaches the call as `ts.OptionalValue` of it. Nothing gave it back: rc held only an array given as it is,
// and own owned only such a one, so every call leaked the literal (2M rounds of `new Set(["a" + i])`:
// 220 MB). Read by optional-array-argument.cmake under rc and own: each array in `reads` is given back
// after its call. It runs in every model.
class Box {
    n: number;
    constructor(values?: string[]) {
        this.n = values ? values.length : 0;
    }
}

function len(values?: string[]): number {
    return values ? values.length : 0;
}

function lenNumbers(values?: number[]): number {
    return values ? values.length : 0;
}

function reads(i: number): number {
    return len(["a" + i, "b"]) + lenNumbers([i, i + 1]) + new Box(["c" + i]).n;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 20000; i++) {
        total += reads(i);
    }

    assert(total == 100000, "total");
    print("done.");
}

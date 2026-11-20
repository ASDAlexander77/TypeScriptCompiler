// `Array.isArray(x)` is `typeof x === "array"`: answered from x's type at compile time, or by a union's or an `any`'s
// tag, and it narrows x in both branches (also after an `if` that returns) - "Class member 'isArray' can't be found".
function len(v: number[] | string): number {
    if (Array.isArray(v)) {
        return v.length * 10;
    }
    return v.length;
}

function first(v: string[] | number): string {
    return Array.isArray(v) ? v[0] : "n" + v;
}

function main() {
    const a = [1, 2, 3];
    assert(Array.isArray(a));
    assert(!Array.isArray(5));
    assert(!Array.isArray("s"));
    assert(len([1, 2]) == 20 && len("abc") == 3);
    assert(first(["x"]) == "x" && first(7) == "n7");
    const x: any = [1];
    const y: any = 1;
    assert(Array.isArray(x) && !Array.isArray(y));
    print("done.");
}

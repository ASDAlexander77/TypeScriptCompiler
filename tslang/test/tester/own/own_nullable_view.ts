// A string or an array widened to a union that also holds null or undefined is the same block, and
// so is one narrowed back out of it: rc's retain of either is one of the block. A number widened
// into a union holds no block, and `ts.SafeCast` - the payload a `typeof` test narrowed - is the
// payload. own takes nothing for any of them.
function nameOr(n: number): string | null {
    if (n > 0) return "named";
    return null;
}

function keepName(s: string): string | null {
    return s;
}

function sure(s: string | null): string {
    return <string>s;
}

function listOf(x: number[]): number[] | null {
    return x;
}

function widen(u: number | string): boolean | number | string {
    if (typeof u == "string") return u;
    return u;
}

function main() {
    const a = nameOr(1);
    const b = nameOr(0);
    assert(a == "named" && b == null);

    let made = "made" + 1;
    const kept = keepName(made);
    assert(kept == "made1");
    assert(sure(kept) == "made1");

    const xs = listOf([1, 2, 3]);
    assert(xs != null && xs.length == 3);

    const w = widen("w");
    const n = widen(2);
    assert(typeof w == "string" && typeof n == "number");

    print("done.");
}

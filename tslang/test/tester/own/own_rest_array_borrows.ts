// The array a rest parameter is given (`total(a, b)`) only borrows its elements when the callee does
// not keep the parameter: rc's references for them go, the arguments stay their owners' and are used
// after the call, and the array is given back after it without them (spec 49). A parameter given on
// to a rest parameter (a callback's `cur`) is borrowed as well. Each step churns, so a value freed
// too early reads garbage; every model runs it.
class C {
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) junk.push(new C(999));
}

function total(...items: C[]) {
    let n: number = 0;
    for (const c of items) n += c.x;
    return n;
}

function lengths(this: int[][], ...other: int[][]) {
    let n = this.length;
    for (const o of other) n += o.length;
    return n;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 300; i++) {
        const a = new C(i);
        const b = new C(1);
        if (total(a, b) == i + 1) ok++;
        churn();
        if (a.x == i && b.x == 1 && total(b, b, a) == i + 2) ok++;

        const xs = [[1, 2]];
        const ys = [3, 4, 5];
        const f = (acc: int[][], cur: int[]) => acc.lengths(cur);
        if (xs.lengths(ys) == 4 && f(xs, ys) == 4) ok++;
        churn();
        if (ys.length == 3 && ys[2] == 5) ok++;
    }

    assert(ok == 300 * 4, "ok");
    print("done.");
}

// A function whose result borrows its parameter (`return this`) still copies a string it reads out
// of that parameter when it stores it into an element, as any function does (spec 41):
// `this[t] = this[s]`, and a swap's `let v = this[i]`. What it returns stays the borrow - read and
// returned, directly or through a local - so the caller does not free it twice, and a copy of it
// would leak. Every string is built at run time (a literal is immortal and would hide a missing copy),
// and each case churns between steps.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

function made(n: int): string[] {
    const r: string[] = [];
    for (let i = 0; i < n; i++) r.push("s" + i);
    return r;
}

function copyWithin(this: string[], target: int, start: int, count: int) {
    for (let i = 0; i < count; i++) this[target + i] = this[start + i];
    return this;
}

function swapEnds(this: string[]) {
    let v = this[0];
    this[0] = this[this.length - 1];
    this[this.length - 1] = v;
    return this;
}

function first(this: string[]) {
    return this[0];
}

function firstOrSecond(this: string[]) {
    let x = this[0];
    if (this.length > 9) x = this[1];
    return x;
}

function main() {
    let ok = 0;
    for (let i = 0; i < 1000; i++) {
        const a = made(4);
        const same = a.copyWithin(0, 2, 2);
        churn();
        if (same[0] == "s2" && a[1] == "s3" && a[2] == "s2" && a[3] == "s3") ok++;

        const b = made(3);
        b.swapEnds();
        churn();
        if (b[0] == "s2" && b[2] == "s0") ok++;

        const f = b.first();
        const g = b.firstOrSecond();
        churn();
        if (f == "s2" && g == "s2") ok++;
    }

    assert(ok == 1000 * (3), "ok");
    print("done.");
}

// A `const`/`let` read off an array element and used after a call that may free that element (`const e = a[0];
// new Q(a); e.v` of a constructor that empties `a`) holds a reference of its own under -mm=own (USER RULING 11-27):
// a copy of a string or an `any`, else a count of the object (of a closure's capture box), given back where it is last
// used on each path - before an early return or a throw, on a `continue` or `break` past it, before a loop reads it
// again. It was an error ("'e' borrows 'a[]' but is used here after it may be released or overwritten").
// Every model runs it; after each free the same-sized blocks are allocated again, so an element freed under `e`
// would read another value.

class B {
    v = 0;
    constructor(v: number) {
        this.v = v;
    }
}

class QB {
    n: number;
    constructor(arr: B[]) {
        arr.pop();
        this.n = 1;
    }
}

class QN {
    n: number;
    constructor(arr: number[][]) {
        arr.pop();
        this.n = 1;
    }
}

class QS {
    n: number;
    constructor(arr: string[]) {
        arr.pop();
        this.n = 1;
    }
}

class QA {
    n: number;
    constructor(arr: any[]) {
        arr.pop();
        this.n = 1;
    }
}

function popAll(a: B[]) {
    while (a.length > 0) a.pop();
}

const sink: B[] = [];
function churn(k: number) {
    for (let i = 0; i < 8; i++) sink.push(new B(-k - i));
    while (sink.length > 4) sink.pop();
}

function churnArrays(k: number) {
    const t: number[][] = [];
    for (let i = 0; i < 8; i++) t.push([-k, -k]);
    return t.length;
}

function churnStrings(k: number) {
    const t: string[] = [];
    for (let i = 0; i < 8; i++) t.push("zz" + (-k));
    return t.length;
}

function objects(k: number) {
    const a = [new B(k)];
    const e = a[0];
    const q = new QB(a);
    churn(k);
    return e.v == k && q.n == 1;
}

function lets(k: number) {
    const a = [new B(k)];
    let e = a[0];
    popAll(a);
    churn(k);
    return e.v == k;
}

function arrays(k: number) {
    const a = [[k, k + 1]];
    const e = a[0];
    const q = new QN(a);
    churnArrays(k);
    return e[0] == k && e[1] == k + 1 && q.n == 1;
}

function strings(k: number) {
    const a = ["s" + k];
    const e = a[0];
    const q = new QS(a);
    churnStrings(k);
    return e == "s" + k && q.n == 1;
}

function early(k: number, stop: boolean) {
    const a = [new B(k)];
    const e = a[0];
    if (stop) return true;
    popAll(a);
    churn(k);
    return e.v == k;
}

// a throw between the call and the last use
function throws(k: number, fail: boolean) {
    const a = [new B(k)];
    const e = a[0];
    popAll(a);
    if (fail) throw "fail";
    churn(k);
    return e.v == k;
}

// a continue and a break between them, the read inside the loop
function loops(n: number) {
    let good = 0;
    for (let k = 0; k < n; k++) {
        const a = [new B(k)];
        const e = a[0];
        popAll(a);
        if (k % 3 == 0) continue;
        if (k == n - 1) break;
        churn(k);
        if (e.v == k) good++;
    }

    return good;
}

// uses in a loop after the call, the read before it
function spread(k: number) {
    const a = [new B(k)];
    const e = a[0];
    popAll(a);
    let t = 0;
    for (let i = 0; i < 3; i++) {
        churn(k + i);
        t += e.v;
    }

    return t == 3 * k;
}

function popFns(fs: (() => number)[]) {
    while (fs.length > 0) fs.pop();
}

// a closure: a count of its capture box
function closures(k: number) {
    const fs: (() => number)[] = [() => k + 1];
    const f = fs[0];
    popFns(fs);
    churn(k);
    return f() == k + 1;
}

// through a class value, of an `any[]` (the case TSLANG 218 still refused)
function make<T>(c: new (...args: any[]) => T, a: any[]): T {
    const e = a[0];
    const q = new c(a);
    churnStrings(1);
    if (e != "m1") throw "make";
    return q;
}

function main() {
    let bad = 0;
    for (let k = 0; k < 2000; k++) {
        if (!objects(k)) bad++;
        if (!lets(k)) bad++;
        if (!arrays(k)) bad++;
        if (!strings(k)) bad++;
        if (!early(k, (k & 1) == 1)) bad++;
        make(QA, ["m" + 1]);
        try {
            if (!throws(k, (k & 1) == 1)) bad++;
        } catch (e) {
        }

        if (!spread(k)) bad++;
        if (!closures(k)) bad++;
    }

    if (loops(50) != 32) bad++;
    if (bad != 0) throw "a borrow outlived its element";
    print("done.");
}

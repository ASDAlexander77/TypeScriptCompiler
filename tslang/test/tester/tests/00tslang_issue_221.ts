// TSLANG 221: under -mm=own, a borrow used after its place may be freed takes a reference of its own (USER RULING
// 11-27, which 219 applied to calls) also where:
// - this function frees it itself: `a.pop()`, `a[0] = x`;
// - the local is assigned again (`let e = a[0]; ...; e = b[0]`), a string's too;
// - a value chosen from it (`k % 2 == 0 ? e : o`) is what is used after;
// - a loop's `continue` or `break` comes between (branches by then, as 219 placed them).
// And the reference given back after a field read through it (`e.v`), a method call on it, an interface view of
// it: before, the count of a `const` went back between the field's address and its load.

class B {
    v = 0;
    constructor(v: number) {
        this.v = v;
    }
}

class C {
    v: number = 0;
    constructor(v: number) {
        this.v = v;
    }
    get() {
        return this.v;
    }
}

interface HasV {
    v: number;
}

function empty(a: B[]) {
    while (a.length > 0) a.pop();
}

function emptyC(a: C[]) {
    while (a.length > 0) a.pop();
}

function emptyS(a: string[]) {
    while (a.length > 0) a.pop();
}

const sink: B[] = [];
function churn(k: number) {
    for (let i = 0; i < 8; i++) sink.push(new B(-k - i));
    while (sink.length > 4) sink.pop();
}

// popped by this function
function popped(k: number) {
    const a = [new B(k), new B(k + 1)];
    const e = a[1];
    a.pop();
    churn(k);
    return e.v == k + 1;
}

// overwritten by this function
function overwritten(k: number) {
    const a = [new B(k), new B(k + 1)];
    const e = a[0];
    a[0] = new B(-1);
    churn(k);
    return e.v == k && a[0].v == -1;
}

// a `let`, popped
function letPopped(k: number) {
    const a = [new B(k), new B(k + 1)];
    let e = a[1];
    a.pop();
    churn(k);
    return e.v == k + 1;
}

// strings, popped and overwritten
function strings(k: number) {
    const a = ["x" + k, "y" + k];
    const e = a[1];
    a.pop();
    const f = a[0];
    a[0] = "w" + k;
    churn(k);
    return e == "y" + k && f == "x" + k && a[0] == "w" + k;
}

// a `let` assigned again
function reassigned(k: number) {
    const a = [new B(k), new B(k + 1)];
    const b = [new B(k + 2)];
    let e = a[0];
    empty(a);
    churn(k);
    const ok = e.v == k;
    e = b[0];
    empty(b);
    churn(k);
    return ok && e.v == k + 2;
}

function reassignedString(k: number) {
    const a = ["x" + k, "y" + k];
    const b = ["z" + k];
    let e = a[0];
    emptyS(a);
    churn(k);
    const ok = e == "x" + k;
    e = b[0];
    emptyS(b);
    churn(k);
    return ok && e == "z" + k;
}

// chosen from
function chosen(k: number) {
    const a = [new B(k), new B(k + 1)];
    const o = new B(7);
    const e = a[0];
    empty(a);
    churn(k);
    const f = k % 2 == 0 ? e : o;
    return f.v == (k % 2 == 0 ? k : 7);
}

// a loop's continue and break in between
function looped(k: number) {
    let ok = true;
    for (let i = 0; i < 3; i++) {
        const a = [new B(k + i), new B(k + 1)];
        const e = a[0];
        empty(a);
        churn(k);
        if (i == 1) continue;
        if (!(e.v == k + i)) ok = false;
        if (i == 2) break;
    }
    return ok;
}

// a method of it, an interface of it
function method(k: number) {
    const a = [new C(k), new C(k + 1)];
    const e = a[0];
    emptyC(a);
    churn(k);
    const g = e.get();
    const h: HasV = e;
    return g == k && h.v == k && e.v == k;
}

function main() {
    let bad = 0;
    for (let k = 0; k < 2000; k++) {
        if (!popped(k)) bad++;
        if (!overwritten(k)) bad++;
        if (!letPopped(k)) bad++;
        if (!strings(k)) bad++;
        if (!reassigned(k)) bad++;
        if (!reassignedString(k)) bad++;
        if (!chosen(k)) bad++;
        if (!looped(k)) bad++;
        if (!method(k)) bad++;
    }

    if (bad != 0) throw "a borrow did not keep its value";
    print("done.");
}

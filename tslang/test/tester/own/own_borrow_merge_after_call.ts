// -mm=own, phase 3: a merge of two reads borrows from each. Something that may destroy them before
// the reads - the container's own constructor, a call - does not reach a use through the merge,
// since each branch reads afresh. A push onto the array a `for...of` walks drops nothing.
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

class H {
    c: C = new C(0);
    d: C = new C(7);
}

class N {
    c: C | null = null;
}

function noop() {
}

function pick(i: number) {
    const h = new H();
    const c = i % 2 == 0 ? h.c : h.d;
    churn();
    return c.x;
}

function either(n: N, m: N) {
    noop();
    const c = n.c ?? m.c;
    if (c) {
        return c.x;
    }

    return 0;
}

function main() {
    let t: number = 0;
    const h = new H();
    for (let i = 0; i < 10000; i++) {
        t += pick(i);
        const m = new N();
        m.c = new C(3);
        t += either(new N(), m);
        const c = i % 3 == 0 ? h.c : h.d;
        t += c.x;
        h.c = new C(1);
        h.d = new C(2);
    }

    const arr: C[] = [new C(1)];
    for (const e of arr) {
        t += e.x;
        if (arr.length < 500) {
            arr.push(new C(1));
        }
    }

    churn();
    assert(t == 82165);
    print("done.");
}

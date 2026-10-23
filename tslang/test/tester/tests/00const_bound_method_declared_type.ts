// A const bound method with a declared type (`const c: () => number = G.items[0].get`) keeps its
// object alive, as one without does. Under rc a const made over a read of a field, an element or a
// global holds a count of its own; the declared type casts the bound method to a hybrid function,
// and that cast hid the read, so the const was folded with no count, and `G = mk()` freed the
// object `c()` still ran on. Each read follows allocations of the same size, so a freed block is
// reused.
class W {
    constructor(public x: number) {}
}

class V {
    x: number;
    w: W;
    a: number[];

    constructor(x: number) {
        this.x = x;
        this.w = new W(x);
        this.a = [x];
    }

    get(): number {
        return this.x;
    }
}

class N {
    v: V;
    items: V[];

    constructor(x: number) {
        this.v = new V(x);
        this.items = [new V(x)];
    }
}

@noinline function mk(i: number): N {
    return new N(i);
}

function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(777));
    }
}

let G: N = mk(2);

interface IN {
    v: V;
    items: V[];
}

@noinline function mv(x: number): V {
    return new V(x);
}

function take(n: N, i: IN) {
    return i.items[0].x;
}

function fieldOfGlobal(x: number) {
    G = mk(x);
    const c: () => number = G.v.get;
    G = mk(0);
    churn();
    return c();
}

function fieldOfLocal(x: number) {
    let n = mk(x);
    const c: () => number = n.v.get;
    n = mk(0);
    churn();
    return c();
}

function local(x: number) {
    let v = new V(x);
    const c: () => number = v.get;
    v = new V(0);
    churn();
    return c();
}

function inLoop(x: number) {
    G = mk(x);
    const c: () => number = G.items[0].get;
    for (let i = 0; i < 2; i++) {
        G = mk(i);
        churn();
    }

    return c();
}

function main() {
    // an element of a global, the fuzzer's shape
    const c: () => number = G.items[0].get;
    let s: number = 0;
    for (let i = 0; i < 1; i++) {
        G = mk(4);
        s += take(mk(5), { v: mv(i + 5), items: [mv(2)] });
    }

    assert(c() == 2, "an element of a global");
    assert(s == 2, "the call's own element");

    assert(fieldOfGlobal(4) == 4, "a field of a global");
    assert(fieldOfLocal(5) == 5, "a field of a local");
    assert(local(6) == 6, "a local");
    assert(inLoop(7) == 7, "the global reassigned in a loop");
    print("done.");
}

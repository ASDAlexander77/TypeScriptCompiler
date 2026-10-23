// An argument read out of a place that a later argument overwrites under another name: `p.n` read
// for one argument, and `G.n = mk()` in the next, where `p` is `G`. Under rc the caller holds an
// argument whose place is overwritten before the call (`f(n, (n = mk(), n))`), but it compared the
// places by name, so it missed one reached through another variable, and the overwrite freed the
// block the callee was given. Each read follows allocations of the same size, so a freed block is
// reused.
class V {
    x: number;
    a: number[];

    constructor(x: number) {
        this.x = x;
        this.a = [x];
    }
}

class N {
    x: number;
    v: V;

    constructor(x: number) {
        this.x = x;
        this.v = new V(x);
    }
}

class H {
    n: N = new N(0);
    items: N[] = [new N(0)];
}

@noinline function mk(x: number): N {
    return new N(x);
}

function churn() {
    const junk: N[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new N(777));
    }
}

const HG = new H();
let G = new H();

function read(a: N, b: number) {
    churn();
    return a.x + a.v.x;
}

// a parameter that is the const global
function fieldOfConstGlobal(p: H, x: number) {
    return read((p.n = mk(x), p.n), (HG.n = mk(0), 0));
}

// a parameter that is the global
function fieldOfGlobal(p: H, x: number) {
    return read((p.n = mk(x), p.n), (G.n = mk(0), 0));
}

// two parameters that are one object
function fieldOfParameter(p: H, q: H, x: number) {
    return read((p.n = mk(x), p.n), (q.n = mk(0), 0));
}

// an element, through another parameter that is the same array
function elementOfParameter(a: N[], b: N[], x: number) {
    return read((a[0] = mk(x), a[0]), (b[0] = mk(0), 0));
}

function main() {
    assert(fieldOfConstGlobal(HG, 3) == 6, "a field of the const global, through a parameter");
    assert(fieldOfGlobal(G, 4) == 8, "a field of the global, through a parameter");
    const h = new H();
    assert(fieldOfParameter(h, h, 5) == 10, "a field, through two parameters");
    assert(elementOfParameter(h.items, h.items, 6) == 12, "an element, through two parameters");
    print("done.");
}

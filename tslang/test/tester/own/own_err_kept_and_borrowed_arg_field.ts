// -mm=own rejects: `f` keeps `p` (stored into `h.n`) and borrows `q`; `f(h, n, n)` gives it the same
// object as both, and `h.n = new N(...)` frees it while `q` is still read. The second `n` is a use
// after the move into the first.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public v: V) {}
}

class H {
    n: N = new N(new V(0));
}

@noinline
function mk(i: number) {
    return new N(new V(i));
}

function churn() {
    const hold: V[] = [];
    for (let i = 0; i < 64; i++) {
        hold.push(new V(777));
    }
}

@noinline
function f(h: H, p: N, q: N) {
    h.n = p;
    h.n = new N(new V(2));
    churn();
    return q.v.x + h.n.v.x;
}

function main() {
    const h = new H();
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const n = mk(i);
        s += f(h, n, n);
    }

    print(s);
    assert(s == 501500);
}

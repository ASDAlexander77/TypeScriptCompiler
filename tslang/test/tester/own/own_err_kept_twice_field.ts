// -mm=own rejects: `f` keeps `p` (stored into `h.n`) and `q` (moved into a `let` given another
// value), and `f(h, n, n)` gives it the same object twice. The second `n` is a use after the move
// into the first.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public v: V) {}
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

class H {
    n: N = new N(new V(0));
}

@noinline
function f(h: H, p: N, q: N) {
    h.n = p;
    let t = q;
    t = new N(new V(0));
    churn();
    return h.n.v.x + t.v.x + 2;
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

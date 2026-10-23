// -mm=own rejects: `p` is kept (stored into `h.n`), and `inner`, read out of `p` before the move, is
// bounded by `h.n` from then on (spec 33); `h.n = new N(...)` frees `p` and `p.v`, and `inner` is
// read after it.
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
function f(h: H, p: N) {
    const inner = p.v;
    h.n = p;
    h.n = new N(new V(2));
    churn();
    return inner.x + h.n.v.x;
}

function main() {
    const h = new H();
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        s += f(h, mk(i));
    }

    print(s);
    assert(s == 501500);
}

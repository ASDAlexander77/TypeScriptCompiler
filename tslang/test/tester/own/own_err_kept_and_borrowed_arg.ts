// -mm=own rejects: `f` keeps `p` (spec 33: it moves into a `let` that is given another value) and
// borrows `q`; `f(n, n)` gives it the same object as both, so the callee frees it under its own
// borrow. The second `n` is a use after the move into the first.
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
function f(p: N, q: N) {
    const inner = q.v;
    let r = p;
    r = new N(new V(2));
    churn();
    return inner.x + r.v.x;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const n = mk(i);
        s += f(n, n);
    }

    print(s);
    assert(s == 501500);
}

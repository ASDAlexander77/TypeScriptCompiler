// -mm=own rejects: `p` moves into the `let r` (spec 33: a parameter read into an owning local that
// is given another value is kept), and `inner`, read out of `p` before the move, is bounded by `r`
// from then on; `r = new N(...)` frees `p` and `p.v`, and `inner` is read after it.
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

@noinline
function f(p: N) {
    const inner = p.v;
    let r = p;
    r = new N(new V(2));
    churn();
    return inner.x + r.v.x;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        s += f(mk(i));
    }

    print(s);
    assert(s == 501500);
}

// -mm=own rejects: `f` keeps both `p` and `q` (each moves into a `let` that is given another
// value, spec 33), and `f(n, n)` gives it the same object twice: two owners, each of which frees it.
// The second `n` is a use after the move into the first.
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
function f(p: N, q: N) {
    let r = p;
    let t = q;
    r = new N(new V(2));
    t = new N(new V(0));
    churn();
    return r.v.x + t.v.x - 2;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const n = mk(i);
        s += f(n, n) + i + 2;
    }

    print(s);
    assert(s == 501500);
}

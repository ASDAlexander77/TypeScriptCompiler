// -mm=own rejects: `f` keeps `p` (it moves into a `let` that is given another value, spec 33), and
// one arm of a conditional in the call's own argument list gives `n` another value after `n` was
// read for `p`: `f(n, i >= 0 ? (n = mk(7), 0) : 0)`. On that arm the assignment releases the old
// object before `f` takes it, so the assignment is a use of `n` after the move, though the other
// arm leaves `n` alone.
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
function f(p: N, z: number) {
    const k = p.v.x;
    churn();
    let r = p;
    r = new N(new V(2));
    churn();
    return r.v.x - 2 + z + k;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        let n = mk(i);
        s += f(n, i >= 0 ? (n = mk(7), 0) : 0) + 2;
    }

    print(s);
    assert(s == 501500);
}

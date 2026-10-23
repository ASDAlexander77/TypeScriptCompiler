// -mm=own rejects: `f` keeps `p` (it moves into a `let` that is given another value, spec 33) and
// borrows `q`, and the call's own argument list gives `n` another value after both reads of it:
// `f(n, n, (n = mk(5), 0))`. The assignment releases the old object before `f` takes it through
// `p` and reads it through `q`, so the assignment is a use of `n` after the move.
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
function f(p: N, q: N, z: number) {
    let r = p;
    r = new N(new V(2));
    churn();
    return q.v.x + r.v.x - 2 + z;
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        let n = mk(i);
        s += f(n, n, (n = mk(5), 0)) + 2;
    }

    print(s);
    assert(s == 501500);
}

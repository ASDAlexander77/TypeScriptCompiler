// -mm=own rejects: as own_err_kept_and_borrowed_arg, with `n` an owning `let`: both arguments of
// `f(n, n)` are reads of its slot, and the one `f` borrows is a use after the move into the other.
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
        let n = mk(i);
        if (i > 2000) {
            n = mk(0);
        }

        s += f(n, n);
    }

    print(s);
    assert(s == 501500);
}

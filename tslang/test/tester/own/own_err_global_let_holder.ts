// -mm=own rejects: the `let` `l` takes the read of `G` out of the object literal `o`, a borrow of
// the block the global owns, and `f` gives `G` a new value before `l` is read, which frees it.
class V {
    a: number[] = [];
    constructor(public x: number) {
        for (let i = 0; i < 64; i++) {
            this.a.push(i);
        }
    }
}

@noinline
function churn() {
    const junk: V[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new V(999));
    }
}

let G = new V(0);

function f(i: number) {
    const o = { v: G };
    let l = o.v;
    G = new V(i + 1);
    churn();
    print(l.x, l.a.length);
}

function main() {
    for (let i = 0; i < 5; i++) {
        f(i);
    }

    print(G.x);
    print("done.");
}

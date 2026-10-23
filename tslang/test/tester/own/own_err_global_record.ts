// -mm=own rejects: the object literal `o` holds the read of `G` - a borrow of the block the global
// owns - and `f` gives `G` a new value before `o.v` is read, which frees that block.
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
    G = new V(i + 1);
    churn();
    print(o.v.x, o.v.a.length);
}

function main() {
    for (let i = 0; i < 5; i++) {
        f(i);
    }

    print(G.x);
    print("done.");
}

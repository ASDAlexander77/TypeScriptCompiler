// -mm=own rejects: `let { v }` takes the read of `K.w` out of the literal - a borrow of the block
// the static owns - and the top-level loop gives `K.w` a new value before `v` is read, which frees
// that block.
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

class K {
    static w: V;
}

for (let i = 0; i < 5; i++) {
    let { v } = { v: K.w };
    K.w = new V(i + 1);
    churn();
    if (i > 0) {
        print(v.x, v.a.length);
    }
}

function main() {
    print(K.w.x);
    print("done.");
}

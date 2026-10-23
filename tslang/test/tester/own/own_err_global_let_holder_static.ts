// -mm=own rejects: the object literal `o` holds the read of `K.w`, and the `let` `l` takes that
// value out of it - a borrow of the block the static owns, still. The static block's loop gives
// `K.w` a new value before `l` is read, which frees that block (spec 25).
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
    static {
        for (let i = 0; i < 5; i++) {
            const o = { v: K.w };
            let l = o.v;
            K.w = new V(i + 1);
            churn();
            if (i > 0) {
                print(l.x, l.a.length);
            }
        }
    }
}

function main() {
    print(K.w.x);
    print("done.");
}

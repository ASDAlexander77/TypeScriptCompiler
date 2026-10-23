// -mm=own rejects: the object literal `o` holds the read of `K.w` - a borrow of the block the static
// owns, not a second owner - and the static block's loop gives `K.w` a new value before `o.v` is
// read, which frees that block: `K.w` is not immortal, its store runs once per round (spec 25).
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
            const c = K.w;
            const o = { v: c };
            K.w = new V(i + 1);
            churn();
            if (i > 0) {
                print(o.v.x, o.v.a.length);
            }
        }
    }
}

function main() {
    print(K.w.x);
    print("done.");
}

// -mm=own rejects: the object literal `o` is declared before the static block's loop, which stores
// the read of `K.w` into `o.v` in one round and reads `o.v` in the later ones, after `K.w = new
// V(i + 1)` freed that block. The read `const c = K.w` is made again each round; `o.v` is not: a
// holder is given a new value only where it is stored (spec 25).
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

class Z {
    static z = new V(42);
}

class K {
    static w: V;
    static {
        const o = { v: Z.z };
        for (let i = 0; i < 5; i++) {
            const c = K.w;
            if (i >= 2) {
                print(o.v.x, o.v.a.length);
            }
            if (i == 1) {
                o.v = c;
            }
            K.w = new V(i + 1);
            churn();
        }
    }
}

function main() {
    print(K.w.x);
    print("done.");
}

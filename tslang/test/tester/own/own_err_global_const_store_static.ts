// -mm=own rejects: `c` names the read of `K.w` itself, a borrow of the block the static owns. The
// static block's loop gives `K.w` a new value before `c` is read, which frees that block: `K.w` is
// not immortal, its store runs once per round (spec 25).
class V {
    a: number[] = [];
    constructor(public x: number) {
        for (let i = 0; i < 64; i++) {
            this.a.push(i);
        }
    }
}

class K {
    static w: V;
    static {
        for (let i = 0; i < 5; i++) {
            const c = K.w;
            K.w = new V(i + 1);
            const z = new V(999);
            if (i > 0) {
                print(c.x, c.a.length);
            }
            z.x = 1000 + i;
        }
    }
}

function main() {
    print(K.w.x);
    print("done.");
}

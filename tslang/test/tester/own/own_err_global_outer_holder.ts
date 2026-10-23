// -mm=own rejects: the object literal `o` is declared before the loop, and each round reads `o.v`
// before it stores the read of `G` there. The next round's `const c = G` is a new read, but `o.v`
// still holds the one stored before `G = new V(i + 1)` freed its block: a holder is given a new
// value only where it is stored, never at the read.
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

let G = new V(0);

function f() {
    const o = { v: Z.z };
    for (let i = 0; i < 4; i++) {
        const c = G;
        print(o.v.x, o.v.a.length);
        o.v = c;
        G = new V(i + 1);
        churn();
    }
}

function main() {
    f();
    print(G.x);
    print("done.");
}

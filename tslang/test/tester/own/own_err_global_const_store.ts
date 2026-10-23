// -mm=own rejects: `c` names the read of `G` itself, a borrow of the block the global owns, and `f`
// gives `G` a new value before `c` is read, which frees that block. (`let c = G` is rejected as a
// borrowing local; a `const` names the read and was not.)
class V {
    a: number[] = [];
    constructor(public x: number) {
        for (let i = 0; i < 64; i++) {
            this.a.push(i);
        }
    }
}

let G = new V(0);

function f(i: number) {
    const c = G;
    G = new V(i + 1);
    const z = new V(999);
    print(c.x, c.a.length);
    z.x = 1000 + i;
}

function main() {
    for (let i = 0; i < 5; i++) {
        f(i);
    }

    print(G.x);
    print("done.");
}

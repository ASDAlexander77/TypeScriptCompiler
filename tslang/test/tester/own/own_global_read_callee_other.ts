// A read of a global is used after calls that give other globals new values, or none: -mm=own accepts
// it, and it reads the global's own value.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }
}

let G = new V(1);
let H = new V(2);

function resetH() {
    H = new V(3);
}

function sum(a: number, b: number) {
    return a + b;
}

function readG() {
    return G.x;
}

function main() {
    const c = G;
    resetH();
    const s = sum(c.x, readG());
    assert(c.x == 1 && c.tag == "v1" && s == 2, "the global's value, read before the calls");
    assert(H.x == 3, "the other global's new value");
    G = new V(4);
    assert(G.x == 4, "the global's new value");
    print("done.");
}

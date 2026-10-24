// -mm=own rejects (spec 36): `r` holds a record that borrows `n` and owns nothing, then a tuple that
// owns a new Node; one release of `r` cannot do right by both.
class Node {
    v = 0;

    constructor(v: number) {
        this.v = v;
    }
}

type Rec = { s: Node; k: number };

function mk(n: Node): Rec {
    return { s: n, k: 1 };
}

function main() {
    const n = new Node(5);
    let r = mk(n);
    print(r.s.v);
    r = { s: new Node(7), k: 2 };
    print(r.s.v);
}

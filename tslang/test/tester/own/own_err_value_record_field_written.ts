// -mm=own rejects (spec 36): a field of a record written through the local that holds it would release
// the value the record borrows (the type literal's field is not readonly).
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
    r.s = new Node(9);
    print(n.v, r.s.v);
}

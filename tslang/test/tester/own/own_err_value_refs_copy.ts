// -mm=own rejects (spec 36): `a` is a copy of a value that holds an owned Node; under single ownership
// the copy is a borrow of `recs[0]`, and overwriting `recs[0]` releases the Node it reads.
class Node {
    v = 0;

    constructor(v: number) {
        this.v = v;
    }
}

@value
interface Rec {
    readonly s: Node;
    readonly k: number;
}

function churn() {
    const keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        keep.push(new Node(77777));
    }

    return keep.length;
}

function main() {
    const recs: Rec[] = [];
    recs.push(makeRec(5));
    const a = recs[0];
    recs[0] = makeRec(6);
    churn();
    print(a.s.v);
}

function makeRec(v: number): Rec {
    const n = new Node(v);
    return { s: n, k: 1 };
}

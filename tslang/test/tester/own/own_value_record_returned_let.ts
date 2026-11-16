// -mm=own: a `@value` record returned with a `let` in it takes the `let`'s value (spec 36 borrows it
// only in a record that does not reach the result): the caller owns the record and what it holds.
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

function make(i: number): Rec {
    let n = new Node(i);
    const r: Rec = { s: n, k: 1 };
    return r;
}

function main() {
    for (let i = 0; i < 200; i++) {
        const r = make(i);
        churn();
        assert(r.s.v == i && r.k == 1, "the record keeps its node");
    }

    print("done.");
}

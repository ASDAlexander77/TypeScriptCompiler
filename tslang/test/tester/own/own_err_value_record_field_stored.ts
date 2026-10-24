// -mm=own rejects (spec 36): the record borrows the `let` `n`, and a class field that outlives `n`
// cannot hold it.
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

class Holder {
    r: Rec | null = null;
}

function fill(h: Holder) {
    let n = new Node(5);
    h.r = { s: n, k: 1 };
}

function main() {
    const h = new Holder();
    fill(h);
    churn();
    print(h.r.s.v);
}

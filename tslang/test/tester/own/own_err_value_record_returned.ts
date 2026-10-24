// -mm=own rejects (spec 36): the record borrows the `let` `n`, which is released when `make` returns,
// so it cannot be returned.
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

function make(): Rec {
    let n = new Node(5);
    const r: Rec = { s: n, k: 1 };
    return r;
}

function main() {
    const r = make();
    churn();
    print(r.s.v);
}

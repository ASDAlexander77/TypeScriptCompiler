// -mm=own rejects (spec 36): the record borrows the value of the `let` `n`, which the assignment
// releases before the record is read.
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

function mk(n: Node): Rec {
    return { s: n, k: 1 };
}

function main() {
    let n = new Node(5);
    const r = mk(n);
    n = new Node(6);
    churn();
    print(r.s.v);
}

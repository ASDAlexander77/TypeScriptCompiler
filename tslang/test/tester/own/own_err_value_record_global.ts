// -mm=own rejects (spec 36): the record borrows the `let` `n`, and a global that outlives `n` cannot
// hold it.
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

let g: Rec | null = null;

function fill() {
    let n = new Node(5);
    g = { s: n, k: 1 };
}

function main() {
    fill();
    churn();
    print(g.s.v);
}

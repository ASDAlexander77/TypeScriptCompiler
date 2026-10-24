// -mm=own rejects (spec 36): a closure that escapes `make` captures the record, which borrows the
// `let` `n` that `make` releases.
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

function make(): () => number {
    let n = new Node(5);
    const r: Rec = { s: n, k: 1 };
    return (): number => r.s.v;
}

function main() {
    const f = make();
    churn();
    print(f());
}

// -mm=own rejects (spec 36): a record that borrows `n` cannot be stored into an array that outlives it.
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
    const recs: Rec[] = [];
    {
        const n = new Node(5);
        recs.push(mk(n));
    }

    churn();
    print(recs[0].s.v);
}

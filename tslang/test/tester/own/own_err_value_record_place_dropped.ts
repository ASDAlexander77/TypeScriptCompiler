// -mm=own rejects (spec 36): the record `o` borrows `p.child`, which `drop` overwrites before `o.s` is
// read through the local that holds it. Before the read in place was followed, this compiled and
// printed 77777 (gc 5).
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

class P {
    child = new Node(5);
}

function drop(p: P) {
    p.child = new Node(6);
    churn();
}

function main() {
    const p = new P();
    const o: Rec = { s: p.child, k: 1 };
    drop(p);
    print(o.s.v);
}

// A node an append-only global keeps as well as a list: popping it off the list leaves the node as
// it is, its `next` included. Under -mm=own the node is immortal, so the list's release frees
// nothing, and the node after it, which the kept node and the list both hold then, is made immortal.
class Node {
    next: Node | null = null;
    constructor(public v: number) {}
}

let KEEP: Node[] = [];

function keep(n: Node) {
    KEEP.push(n);
}

class List {
    head: Node | null = null;

    shift() {
        if (this.head) this.head = this.head.next;
    }

    sum() {
        let s = 0.0;
        for (let c = this.head; c; c = c.next) s += c.v;
        return s;
    }
}

function build(l: List, i: number) {
    const a = new Node(i);
    const b = new Node(i + 1);
    b.next = new Node(i + 2);
    a.next = b;
    keep(a);
    l.head = KEEP[KEEP.length - 1];
}

function main() {
    let checks = 0.0;
    for (let i = 0; i < 500; i++) {
        const l = new List();
        build(l, i);
        assert(l.sum() == 3 * i + 3, "built");

        l.shift();      // the kept node leaves the list, and keeps its `next`
        const kept = KEEP[i];
        assert(kept.v == i && kept.next != null && kept.next.v == i + 1, "kept");
        assert(l.sum() == 2 * i + 3, "shifted");

        l.shift();      // a node the kept node still holds
        assert(kept.next != null && kept.next.next != null && kept.next.next.v == i + 2, "kept chain");
        assert(l.sum() == i + 2, "shifted twice");

        const fresh = new Node(1000);
        checks += fresh.v + l.sum();
    }

    let all = 0.0;
    for (const n of KEEP) {
        for (let c: Node | null = n; c; c = c.next) all += c.v;
    }

    assert(all == 3 * 124750 + 1500, "all");
    assert(checks == 500000 + 124750 + 1000, "checks");
    print("done.");
}

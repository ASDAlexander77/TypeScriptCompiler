// A method that returns the object it is called on - `return this` on every path, its result type
// `this` or the class - called on a temporary gives that object back to the caller, who owns it
// (spec 78): a chain from a new value can be returned, kept in a field or an array, or given to a
// parameter that keeps it, as under every other model. On an object someone owns, the result stays
// a borrow of it. Each step churns, so anything freed too early reads garbage; every model runs it.
class Junk {
    v: number[] = [];
    constructor(public n: number) {}
}

function churn() {
    const junk: Junk[] = [];
    for (let i = 0; i < 64; i++) {
        const j = new Junk(999);
        j.v.push(99);
        junk.push(j);
    }
}

class Node {
    items: Node[] = [];
    constructor(public n: number) {}

    static make(n: number): Node {
        return new Node(n);
    }

    // the result type is the class: the body shows the result is `this`
    add(child: Node): Node {
        this.items.push(child);
        return this;
    }

    // the result type `this` says so
    with(n: number): this {
        this.items.push(new Node(n));
        return this;
    }

    total(): number {
        let t = this.n;
        for (const c of this.items) t += c.total();
        return t;
    }
}

class Holder {
    node: Node;
    constructor() {
        this.node = Node.make(0);
    }
}

function tree(): Node {
    return Node.make(1).add(Node.make(2)).with(3);
}

function main() {
    const a = Node.make(10).add(Node.make(1));
    churn();
    assert(a.total() == 11, "kept in a local");

    const t = tree();
    churn();
    assert(t.total() == 6, "returned");

    const b = Node.make(100);
    b.add(Node.make(1).add(Node.make(2)).with(3));
    churn();
    assert(b.total() == 106, "given to a parameter that keeps it");

    // on an object someone owns: a borrow of it, used in place
    b.with(4).with(5);
    churn();
    assert(b.total() == 115, "chained on an owned object");

    const h = new Holder();
    h.node = Node.make(7).with(1);
    churn();
    assert(h.node.total() == 8, "stored into a field");

    const list: Node[] = [];
    list.push(new Node(20).with(2));
    churn();
    assert(list[0].total() == 22, "pushed into an array");

    print("done.");
}

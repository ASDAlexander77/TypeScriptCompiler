// A borrow of the object a WeakRef refers to, used while nothing can free it: a call into a @final
// class that frees nothing keeps it.
@final
class Node {
    parent: WeakRef<Node> | null = null;
    constructor(public v: number) {}

    depth(): number {
        const p = this.parent ? this.parent.deref() : undefined;
        if (p) return p.v + p.depth();
        return 0;
    }
}

function main() {
    const a = new Node(1);
    const b = new Node(2);
    const c = new Node(3);
    b.parent = new WeakRef(a);
    c.parent = new WeakRef(b);
    assert(c.depth() == 3, "depth");
    print("done.");
}

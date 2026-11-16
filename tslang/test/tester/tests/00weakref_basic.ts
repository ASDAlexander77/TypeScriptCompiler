// WeakRef<T>: a child refers to its parent without owning it. deref() gives the parent while it lives.
class Node {
    parent: WeakRef<Node> | null = null;
    children: Node[] = [];
    constructor(public v: number) {}

    add(c: Node) {
        c.parent = new WeakRef(this);
        this.children.push(c);
    }

    depth(): number {
        const p = this.parent ? this.parent.deref() : undefined;
        return p ? 1 + p.depth() : 0;
    }
}

function main() {
    const root = new Node(1);
    const mid = new Node(2);
    const leaf = new Node(3);
    mid.add(leaf);
    root.add(mid);
    assert(root.children[0].children[0].depth() == 2, "depth");
    const w = new WeakRef(root);
    assert(w.deref() === root, "deref");
    print("done.");
}

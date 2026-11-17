// @weak fields (own spec 93): a back-reference read and written as a plain field
class Node {
    @weak parent: Node | null = null;
    children: Node[] = [];
    constructor(public v: number) {}

    add(c: Node) {
        c.parent = this;
        this.children.push(c);
    }

    depth(): number {
        return this.parent ? 1 + this.parent.depth() : 0;
    }
}

class Leaf extends Node {
    constructor(v: number) { super(v); }
    up(): number { return this.parent ? this.parent.v : -1; }
}

class TreeNode<T> {
    @weak parent: TreeNode<T> | null = null;
    constructor(public value: T) {}
}

class Opt {
    @weak up?: Node;
    constructor(public v: number) {}
}

function churn(n: number) {
    let total: number = 0;
    for (let i = 0; i < n; i++) {
        const t = new Node(i);
        total += t.v;
    }
    return total;
}

function tree() {
    const root = new Node(1);
    const mid = new Node(2);
    const leaf = new Node(3);
    mid.parent = root;
    leaf.parent = mid;
    churn(1000);
    if (leaf.depth() != 2) throw "depth";
    if (leaf.parent.parent.v != root.v) throw "chain";
    if (leaf.parent?.parent?.v != 1) throw "optional chain";
    print("tree");

    // move: a child leaves one parent's array for another's
    const a = new Node(10);
    const b = new Node(20);
    a.add(new Node(30));
    const moved = a.children.pop();
    if (moved) b.add(moved);
    churn(1000);
    const c = b.children[0];
    if (c.parent !== b) throw "move";
    if (c.parent.v != 20) throw "move value";
    if (a.children.length != 0) throw "move source";
    print("move");
}

// a subtree freed when this returns, while `keep` still points into it
function dropSubtree(keep: Node) {
    const root = new Node(1);
    root.add(new Node(2));
    keep.parent = root;
    if (keep.parent.v != 1) throw "drop before";
}

function drop() {
    const keep = new Node(9);
    dropSubtree(keep);
    churn(1000);
    // the parent is gone: the link is cleared without reading it
    keep.parent = null;
    if (keep.parent !== null) throw "drop cleared";
    print("drop");
}

function self() {
    const n = new Node(7);
    n.parent = n;
    if (n.parent.v != 7) throw "self";
    print("self");
}

function subclass() {
    const root = new Node(5);
    const l = new Leaf(6);
    l.parent = root;
    churn(1000);
    if (l.up() != 5) throw "subclass up";
    const asBase: Node = l;
    if (asBase.parent.v != 5) throw "subclass base";
    print("subclass");
}

function generic() {
    const a = new TreeNode<number>(1);
    const b = new TreeNode<number>(2);
    b.parent = a;
    const s = new TreeNode<string>("x");
    const t = new TreeNode<string>("y");
    t.parent = s;
    if (b.parent.value != 1) throw "generic number";
    if (t.parent.value != "x") throw "generic string";
    print("generic");
}

function optional() {
    const o = new Opt(1);
    if (o.up !== undefined) throw "optional unset";
    const n = new Node(8);
    o.up = n;
    if (o.up.v != 8) throw "optional set";
    print("optional");
}

function main() {
    tree();
    drop();
    self();
    subclass();
    generic();
    optional();
    print("done.");
}

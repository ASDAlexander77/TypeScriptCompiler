// own spec 93: each read of a @weak field is checked again, so reading it before and after a call
// that may free things compiles; a call that frees nothing may run on a @weak read.
class Node {
    @weak parent: Node | null = null;
    v: number;
    constructor(v: number) { this.v = v; }
    depth(): number { return this.parent ? 1 + this.parent.depth() : 0; }
}
class Holder { n: Node | null = null; }
function dropAll(h: Holder) { h.n = null; }
function main() {
    const root = new Node(1);
    const mid = new Node(2);
    const leaf = new Node(3);
    mid.parent = root;
    leaf.parent = mid;
    const h = new Holder();
    h.n = new Node(4);
    let total: number = 0;
    total += leaf.parent ? leaf.parent.v : 0;
    dropAll(h);
    total += leaf.parent ? leaf.parent.v : 0;
    total += leaf.parent ? leaf.parent.depth() : 0;
    if (total != 5) throw "total";
    print("done.");
}

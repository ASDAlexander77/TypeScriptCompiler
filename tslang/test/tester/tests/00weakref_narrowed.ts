// A target whose declared type allows null, narrowed by its assignment (`let n: Node | null = new
// Node()`): `new WeakRef(n)` makes a WeakRef<Node>, as TypeScript reads n there. A null target makes one
// whose deref() gives undefined.
class Node {
    constructor(public v: number) {}
}

function main() {
    let n: Node | null = new Node(42);
    const w = new WeakRef(n);
    const u: WeakRef<Node> | string = w;
    assert(w.deref()?.v == 42, "deref");
    assert(typeof u != "string", "a WeakRef");
    // a null one refers to nothing
    let m: Node | null = null;
    const none = new WeakRef(m);
    assert(none.deref() === undefined, "null target");
    print("done.");
}

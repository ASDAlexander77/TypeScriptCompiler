// An object made in a function and referred to by a WeakRef that outlives the call stays on the heap:
// on the stack its frame would be gone while the WeakRef still reads it. Under -mm=own the object is
// freed when the function ends and deref() gives undefined; under gc and none it may still be found,
// and then it is the object as it was.
class Node {
    constructor(public v: number) {}
}

class Holder {
    w: WeakRef<Node> | null = null;
}

function link(h: Holder, n: Node) {
    h.w = new WeakRef(n);
}

// the WeakRef is made by a function the object is only lent to
function make(h: Holder, v: number) {
    const n = new Node(v);
    link(h, n);
    return n.v;
}

function scribble(v: number) {
    // a frame of the same size over the old one
    const a = new Node(v);
    const b = new Node(v);
    return a.v + b.v;
}

function freesAtOnce() {
    return Shared.count(new Shared(0)) >= 0;
}

function main() {
    const own = freesAtOnce();
    const h = new Holder();
    let s = 0.0;
    for (let i = 0; i < 1000; i++) {
        s += make(h, 5);
        s += scribble(77);
        const n = h.w ? h.w.deref() : undefined;
        if (own) {
            assert(!n, "freed");
        } else if (n) {
            assert(n.v == 5, "intact");
        }
    }

    assert(s == 1000 * (5 + 154.0), "sum");
    print("done.");
}

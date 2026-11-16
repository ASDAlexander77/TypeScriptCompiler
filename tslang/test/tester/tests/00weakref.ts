// WeakRef<T> in the shapes that matter: a tree with parent links, a subtree dropped while a WeakRef
// into it remains, copies of one WeakRef, a WeakRef given another target after its own died, an object
// that refers to itself, WeakRefs held by an array, and a WeakRef to an object that never dies. Under
// -mm=own the count is kept in the target's header word and a freed target stays a tombstone until
// the last WeakRef to it goes; gc frees when its collector runs and none never, so there a WeakRef may
// still find what it refers to. New objects of the same size are made between the steps: a freed
// block read by mistake would show.
class Node {
    parent: WeakRef<Node> | null = null;
    children: Node[] = [];
    me: WeakRef<Node> | null = null;
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

class Holder {
    w: WeakRef<Node> | null = null;
    other: WeakRef<Node> | null = null;
}

const G = new Node(7);

// -mm=own frees an object when its owner gives it up (`root.children = []` here): only there is a
// dropped target certainly gone
function freesAtOnce() {
    return Shared.count(new Shared(0)) >= 0;
}

function alive(w: WeakRef<Node> | null) {
    if (!w) return -1;
    const n = w.deref();
    return n ? n.v : -1;
}

function churn() {
    let s = 0.0;
    for (let i = 0; i < 4; i++) {
        const n = new Node(1000);
        s += n.v;
    }

    return s;
}

function main() {
    const FREES_AT_ONCE = freesAtOnce();
    let checks = 0.0;
    for (let i = 0; i < 2000; i++) {
        // tree
        const root = new Node(1);
        const a = new Node(2);
        const b = new Node(3);
        const leaf = new Node(4);
        b.add(leaf);
        a.add(b);
        root.add(a);
        assert(root.children[0].children[0].children[0].depth() == 3, "tree");

        // dropped: the subtree goes (`a`, `b` and `leaf`), the WeakRef into it stays
        const h = new Holder();
        h.w = new WeakRef(root.children[0].children[0]);
        assert(alive(h.w) == 3, "before drop");
        root.children = [];
        checks += churn();
        assert(!FREES_AT_ONCE || alive(h.w) == -1, "after drop");

        // copies: two WeakRefs to one target, the target goes, both see it gone
        const t = new Node(5);
        h.w = new WeakRef(t);
        h.other = h.w;
        assert(alive(h.w) == 5 && alive(h.other) == 5, "copies alive");
        root.children.push(t);
        root.children = [];
        checks += churn();
        assert(!FREES_AT_ONCE || (alive(h.w) == -1 && alive(h.other) == -1), "copies dead");

        // retarget: a dead WeakRef's field given another target
        h.w = new WeakRef(root);
        assert(alive(h.w) == 1, "retarget");
        h.other = null;

        // self: an object that refers to itself, freed
        const s = new Node(6);
        s.me = new WeakRef(s);
        root.children.push(s);
        root.children = [];
        checks += churn();

        // elements: WeakRefs held by an array, popped, spliced, dropped
        const refs: WeakRef<Node>[] = [];
        for (let j = 0; j < 4; j++) refs.push(new WeakRef(root));
        refs.pop();
        refs.splice(0, 1);
        assert(refs.length == 2 && alive(refs[0]) == 1, "elements");

        // immortal: a global's object
        h.other = new WeakRef(G);
        checks += churn();
        assert(alive(h.other) == 7, "immortal");

        checks += root.v;
    }

    assert(checks == 2000 * (4 * 4000.0 + 1), "checks");
    print("done.");
}

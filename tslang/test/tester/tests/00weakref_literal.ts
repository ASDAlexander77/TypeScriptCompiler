// A WeakRef (or a Shared) made in an object literal or a tuple is counted once: the record holds it,
// and its release gives the count back once. Counted twice down, the target was freed while a WeakRef
// still referred to it, and deref() gave a freed block.
class Node {
    pad1 = 1.0;
    pad2 = 2.0;
    constructor(public v: number) {}
}

class Holder {
    n: Node | null = null;
}

function freesAtOnce() {
    return Shared.count(new Shared(0)) >= 0;
}

function inLiteral(n: Node): number {
    const o = { w: new WeakRef(n), x: 1 };
    return o.x;
}

function inTuple(n: Node): number {
    const t: [WeakRef<Node>, number] = [new WeakRef(n), 2];
    return t[1];
}

function returned(n: Node) {
    return { w: new WeakRef(n), x: 3 };
}

function check(kind: number) {
    const h = new Holder();
    const n = new Node(42);
    const w0 = new WeakRef(n);
    h.n = n;
    assert((kind == 0 ? inLiteral(n) : kind == 1 ? inTuple(n) : returned(n).x) > 0, "made");
    assert(w0.deref()?.v == 42, "alive while held");
    h.n = null;
    const n2 = new Node(77);
    const n3 = new Node(88);
    if (freesAtOnce()) {
        assert(w0.deref() === undefined, "gone once its owner freed it");
    }

    return n2.v + n3.v;
}

function main() {
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        total += check(0);
        total += check(1);
        total += check(2);

        const n = new Node(1);
        const o = { w: new WeakRef(n), x: 1 };
        const t: [WeakRef<Node>, number] = [new WeakRef(n), 2];
        total += o.x + t[1] + (o.w.deref()?.v ?? 0) + (t[0].deref()?.v ?? 0);

        const s = { s: new Shared(i), x: 1 };
        if (freesAtOnce()) {
            assert(Shared.count(s.s) == 1, "one count in a literal");
        }

        total += s.x;
    }

    assert(total == 1000 * (3 * 165 + 6), "total");
    print("done.");
}

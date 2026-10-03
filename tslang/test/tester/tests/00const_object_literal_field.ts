// A `const` object literal keeps what its fields hold (#460). Under rc the const was folded into a
// read of the literal's temporary slot, which took no reference for a field read out of another
// object (`{ s: p.child }`), so overwriting `p.child` freed what the literal still held. Each field
// is read after allocations of the same size have reused anything freed.
class Node {
    v = 0;
    name = "";
}

class P {
    child: Node = new Node();
}

function churn() {
    let keep: Node[] = [];
    for (let i = 0; i < 1000; i++) {
        const n = new Node();
        n.v = -i;
        n.name = "junk" + i;
        keep.push(n);
    }

    return keep.length;
}

function main() {
    const p = new P();
    p.child.name = "first" + 1;
    const o = { s: p.child, n: 2 };
    p.child = new Node();
    assert(churn() == 1000);
    assert(o.s.name == "first1" && o.n == 2, "a const literal keeps the field it read");

    let a = new Node();
    a.name = "local" + 3;
    const t = { x: a, inner: { y: a } };
    a = new Node();
    assert(churn() == 1000);
    assert(t.x.name == "local3" && t.inner.y.name == "local3", "a const literal keeps a local it read");

    for (let i = 0; i < 1000; i++) {
        const q = new P();
        const held = { c: q.child };
        q.child = new Node();
        held.c.v = i;
    }

    print("done.");
}

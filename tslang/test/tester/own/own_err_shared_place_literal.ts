// Shared<T> (spec 23.3, Ruling 7): one read of a handle out of a field with two uses. The store into
// `q.child` gives that use a counted owner of its own (rc retains for it); the object literal's
// field does not (rc retains nothing for it), so that store keeps a borrow of `p.child`. `dropBoth`
// overwrites both fields and gives back the last count of the block the literal points to:
// rejected. A store counts only when rc retains for it; the counted store does not exempt the other.
class Node {
    v = 0;
    tag = "";
}

class P {
    child: Shared<Node> = new Shared(new Node());
}

function churn() {
    let keep: string[] = [];
    for (let i = 0; i < 1000; i++) keep.push("k" + i);
    return keep.length;
}

function dropBoth(p: P, q: P) {
    p.child = new Shared(new Node());
    q.child = new Shared(new Node());
    churn();
}

function main() {
    const p = new P();
    const q = new P();
    p.child.value.tag = "alive";
    const o = { s: (q.child = p.child) };
    dropBoth(p, q);
    print(Shared.count(o.s));
    print(o.s.value.tag);
    print("done.");
}

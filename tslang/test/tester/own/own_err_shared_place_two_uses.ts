// Shared<T> (spec 23.3, Ruling 7): one read of a handle out of a field with two uses. The store into
// `q.child` gives that use a counted owner of its own; the argument given to `f` does not, so it
// is a borrow of `p.child`. `f` overwrites both fields, which may give back the last count of the
// block the argument points to: rejected. The counted use does not exempt the other one.
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

function f(p: P, q: P, s: Shared<Node>) {
    p.child = new Shared(new Node());
    q.child = new Shared(new Node());
    churn();
    print(Shared.count(s));
    print(s.value.tag);
}

function main() {
    const p = new P();
    const q = new P();
    p.child.value.tag = "alive";
    f(p, q, q.child = p.child);
    print("done.");
}

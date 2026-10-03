// Shared<T> (spec 23.3, Rulings 7 and 9): one read of a handle out of a field with two uses. The
// store into `q.child` gives that use a counted owner of its own (rc retains for it); the tuple's
// element does not (rc retains nothing for it, and a tuple is no receiver rc counts), so the tuple
// keeps a borrow of `p.child`. `dropBoth` overwrites both fields and gives back the last count of
// the block the tuple points to: rejected. A use that is not known to count is a borrow (Ruling 9).
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
    const t: [Shared<Node>, number] = [(q.child = p.child), 1];
    dropBoth(p, q);
    print(Shared.count(t[0]));
    print(t[0].value.tag);
    print("done.");
}

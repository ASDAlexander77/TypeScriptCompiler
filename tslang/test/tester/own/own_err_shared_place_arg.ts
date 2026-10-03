// Shared<T> (spec 23.3): a handle read out of a field and given straight to a call is a borrow of
// the field, not a counted copy. The callee overwrites the field, which may give back the last
// count of the block the argument points to: rejected, as for a class read out of a field.
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

function f(p: P, s: Shared<Node>) {
    p.child = new Shared(new Node());
    churn();
    print(Shared.count(s));
    print(s.value.tag);
}

function main() {
    const p = new P();
    p.child.value.tag = "alive";
    f(p, p.child);
    print("done.");
}

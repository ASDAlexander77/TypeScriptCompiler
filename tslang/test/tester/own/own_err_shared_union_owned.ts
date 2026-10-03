// Shared<T> (spec 23.2) in a union with a member that owns a block of its own: -mm=own counts a
// handle and owns a string singly, so it cannot do both for one value. Rejected under own only.
class Node {
    v = 0;
}

function main() {
    const a = new Shared(new Node());
    let u: Shared<Node> | string = a;
    u = "text";
    print(Shared.count(a));
}

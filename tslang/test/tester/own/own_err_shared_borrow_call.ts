// -mm=own rejects (spec 23.3): a borrow read through a handle, used after a call that may drop -
// `touch` writes through a handle it reaches from a global.
class Node {
    v = 0;
}

let kept: Shared<Node> | null = null;

function touch() {
    const k = kept;
    if (k) k.value = new Node();
}

function main() {
    const a = new Shared(new Node());
    kept = a;
    const n = a.value;
    touch();
    print(n.v);
}

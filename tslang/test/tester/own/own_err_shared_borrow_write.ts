// -mm=own rejects (spec 23.3): a borrow read through one handle, used after a write through another
// handle that may reach the same block.
class Node {
    v = 0;
}

function main() {
    const a = new Shared(new Node());
    const b = a;
    const n = a.value;
    b.value = new Node();
    print(n.v);
}

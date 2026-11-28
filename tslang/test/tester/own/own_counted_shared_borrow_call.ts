// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
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

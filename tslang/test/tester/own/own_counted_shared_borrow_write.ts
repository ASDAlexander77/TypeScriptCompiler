// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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

// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// -mm=own rejects (spec 23.3): a borrow read through a handle, used after that handle is given a
// new block - the old one may have been the last handle, and its payload went with it.
class Node {
    v = 0;
}

function main() {
    let a = new Shared(new Node());
    const n = a.value;
    a = new Shared(new Node());
    print(n.v);
}

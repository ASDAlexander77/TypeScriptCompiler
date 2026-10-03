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

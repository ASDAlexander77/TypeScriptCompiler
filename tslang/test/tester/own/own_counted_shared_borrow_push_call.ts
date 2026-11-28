// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own rejects (spec 23.3): an element borrowed through a handle, used after a call that pushes
// through another handle to the same block, which may move the elements.
class Node {
    v = 0;
}

class Bag {
    items: Node[] = [];
}

function grow(s: Shared<Bag>) {
    s.value.items.push(new Node());
}

function main() {
    const a = new Shared(new Bag());
    a.value.items.push(new Node());
    const first = a.value.items[0];
    grow(a);
    print(first.v);
}

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

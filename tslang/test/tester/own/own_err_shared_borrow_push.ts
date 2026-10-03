// -mm=own rejects (spec 23.3): an element borrowed through one handle, used after a push through
// another, which may move the elements.
class Node {
    v = 0;
}

class Bag {
    items: Node[] = [];
}

function main() {
    const a = new Shared(new Bag());
    const b = a;
    a.value.items.push(new Node());
    const first = a.value.items[0];
    b.value.items.push(new Node());
    print(first.v);
}

// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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

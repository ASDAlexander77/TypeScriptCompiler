// Shared<T> (spec 23.4): a doubly linked pair and a child with a link to its parent are cycles,
// which are never freed (as under rc); the program still runs.
class Item {
    v = 0;
    next: Shared<Item> | null = null;
    prev: Shared<Item> | null = null;
}

function main() {
    for (let i = 0; i < 100; i++) {
        const a = new Shared(new Item());
        const b = new Shared(new Item());
        a.value.next = b;
        b.value.prev = a;
        a.value.v = i;
        const p = b.value.prev;
        assert(p !== null && p.value.v == i);

        const parent = new Shared(new Item());
        const child = new Shared(new Item());
        parent.value.next = child;
        child.value.prev = parent;
    }

    print("done.");
}

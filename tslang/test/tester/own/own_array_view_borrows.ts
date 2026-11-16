// `a.view(from, to)` copies the element pointers of a slice of `a`. Under -mm=own a view that is only
// read - its length and its elements - borrows them (spec 91): the copy is given back without what it
// holds, and the elements stay the source's. A view of a parameter, of a local and of a field.
// Values are made at run time, in a loop, so a double release or a leak shows.
class Item {
    constructor(public v: number, public name: string) {}
}

class Holder {
    items: Item[] = [];
}

function peek(a: Item[]) {
    const w = a.view(0, 1);
    assert(w.length == 2, "view(0, 1) has two elements");
    return w[0].v + w[1].v;
}

function names(a: string[]) {
    const w = a.view(1, 1);
    return w[0].length;
}

function main() {
    const h = new Holder();
    let total = 0;
    for (let i = 0; i < 3000; i++) {
        const items: Item[] = [new Item(i, "a" + i), new Item(i + 1, "b" + i), new Item(i + 2, "c" + i)];
        total += peek(items);

        const local = items.view(1, 40);
        assert(local.length == 2 && local[0].v == i + 1 && local[1].name == "c" + i, "a view of a local, clamped");

        h.items.push(new Item(i, "h" + i));
        if (h.items.length > 4) {
            h.items.shift();
        }

        const field = h.items.view(0, 0);
        total += field[0].v;

        const texts = ["x" + i, "yy" + i];
        total += names(texts);

        assert(items[0].name == "a" + i && items[2].name == "c" + i, "the source keeps its elements");
    }

    assert(total > 0, "total");
    print("done.");
}

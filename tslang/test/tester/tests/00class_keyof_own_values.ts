// -mm=own: heap values through a keyof switch, in a loop - a run-time string and an object written (the value
// moves into whichever field the key names: one store in each branch) and read back (borrowed).
class Item {
    constructor(public n: number) {}
}

class Holder {
    s: string = "";
    t: string = "";
    item: Item = new Item(0);
    other: Item = new Item(0);
}

function strKey(i: number): "s" | "t" {
    return i % 2 == 0 ? "s" : "t";
}

function itemKey(i: number): "item" | "other" {
    return i % 2 == 0 ? "item" : "other";
}

function main() {
    let total = 0;
    for (let i = 0; i < 2000; i++) {
        const h = new Holder();
        h[strKey(i)] = "v" + i;
        assert(h[strKey(i)] == "v" + i);

        h[itemKey(i)] = new Item(i);
        total += h[itemKey(i)].n;
    }

    assert(total == 1999 * 1000);
    print("done.");
}

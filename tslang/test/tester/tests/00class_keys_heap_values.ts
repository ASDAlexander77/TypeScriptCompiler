// Heap values through a class's keys, in a loop: strings made at run time ("s" + i, not literals, which are
// immortal) and objects, read through a keyof switch and a string key, written through a keyof switch, and a
// method read through any after the object is only in the any. Under rc a missing retain shows as a freed value
// read (a wrong string or a crash), a missing release as a leak.
class Item {
    constructor(public n: number) {}
}

class Holder {
    s: string = "";
    t: string = "";
    item: Item = new Item(0);
    other: Item = new Item(0);

    name(): string {
        return "h" + this.s;
    }
}

function strKey(i: number): "s" | "t" {
    return i % 2 == 0 ? "s" : "t";
}

function itemKey(i: number): "item" | "other" {
    return i % 2 == 0 ? "item" : "other";
}

function key(s: string) {
    return s;
}

function main() {
    let total = 0;
    for (let i = 0; i < 2000; i++) {
        const h = new Holder();

        // a run-time string written and read through a keyof switch
        h[strKey(i)] = "v" + i;
        const read = h[strKey(i)];
        assert(read == "v" + i);

        // an object written through the switch: one value, a store in each branch
        h[itemKey(i)] = new Item(i);
        const item = h[itemKey(i)];
        assert(item.n == i);
        total += item.n;

        // read through a string key, as any
        const viaString = <string>h[key(i % 2 == 0 ? "s" : "t")];
        assert(viaString == "v" + i);

        // a method read through any, called after the any is the only one holding the object
        const f = <() => string>(<any>new Holder())[key("name")];
        assert(f() == "h");
    }

    assert(total == 1999 * 1000);
    print("done.");
}

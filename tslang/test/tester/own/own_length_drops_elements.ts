// -mm=own: a method that sets an array's length (or pops, shifts, splices it) may free the elements of
// arrays of that element type, and nothing else. A borrow of an element of another type lent to it
// outlives the call: `grow` here cannot free a string out of `names`.
class Entry {
    constructor(public key: string, public n: number) {}
}

class Table {
    entries: Entry[] = [];

    grow(size: number) {
        this.entries.length = size;
    }

    put(key: string, n: number) {
        this.grow(this.entries.length + 1);
        this.entries[this.entries.length - 1] = new Entry(key, n);
    }

    fill(names: string[]) {
        for (let i = 0; i < names.length; i++) this.put(names[i], i);
    }
}

function main() {
    const t = new Table();
    t.fill(["a" + 1, "b" + 2]);
    assert(t.entries.length == 2 && t.entries[0].key == "a1" && t.entries[1].n == 1);
    print("done.");
}

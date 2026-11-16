// A callback that declares fewer parameters than its type (`(v) => ...` for `(value, key, map) => void`)
// is called through an adapter that passes on what it is given. A callback that keeps a string it is
// lent (`names.push(k)`) gets a copy, as one that declares every parameter does: Map and Set's forEach
// lend each entry's key and value out of their entries.
class Entry {
    constructor(public key: string, public value: string) {}
}

class Table {
    entries: Entry[] = [];
    tuples: [string, number][] = [];

    add(key: string, value: string, tupleKey: string) {
        this.entries.push(new Entry(key, value));
        this.tuples.push([tupleKey, 1]);
    }

    forEach(f: (value: string, key: string, table: Table) => void) {
        for (let i = 0; i < this.entries.length; i++) {
            f(this.entries[i].value, this.entries[i].key, this);
        }
    }

    forEachTuple(f: (key: string, count: number, table: Table) => void) {
        for (let i = 0; i < this.tuples.length; i++) {
            f(this.tuples[i][0], this.tuples[i][1], this);
        }
    }
}

function run() {
    const t = new Table();
    t.add("a", "x", "ta");
    t.add("b", "y", "tb");

    const values: string[] = [];
    t.forEach((v) => { values.push(v); });
    const keys: string[] = [];
    t.forEach((v, k) => { keys.push(k); });
    const fromTuples: string[] = [];
    t.forEachTuple((k) => { fromTuples.push(k); });

    t.add("c", "z", "tc");
    assert(values.length == 2 && values[0] == "x" && values[1] == "y", "values");
    assert(keys.length == 2 && keys[0] == "a" && keys[1] == "b", "keys");
    assert(fromTuples.length == 2 && fromTuples[1] == "tb", "tuple keys");
}

function main() {
    for (let i = 0; i < 200; i++) {
        run();
    }

    print("done.");
}

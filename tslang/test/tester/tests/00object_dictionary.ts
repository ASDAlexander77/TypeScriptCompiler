// A string index signature, `{ [key: string]: T }`, is a dictionary: the default library's ObjectDictionary<T>,
// whose entries are known at run time. `d[k]`, `d["name"]` and `d.name` are its entries - never its members, so
// `d.get` and `d.size` are entries too - read as `T | undefined` (undefined for a key it does not hold) and
// written as T. An object literal with a key known only at run time (`{ [m]: i }`) makes one, and so does one
// given where a dictionary is wanted. `k in d`, `for...in`, `Object.keys` and `delete d[k]` go by its entries.
// Under -mm=own it owns its values: written in moves, read borrows, overwritten or deleted drops.
// The tests compile without the default library: its ObjectDictionary (and objectDictionaryAssign), as the
// compiler finds them by name, over two arrays.
@dictionary
class ObjectDictionary<T = any> {
    private _keys: string[] = [];
    private _values: T[] = [];

    get size() {
        return this._keys.length;
    }

    private find(key: string): int {
        for (let i = 0; i < this._keys.length; i++) {
            if (this._keys[i] == key) {
                return i;
            }
        }

        return -1;
    }

    get(key: string): T | undefined {
        const i = this.find(key);
        if (i < 0) {
            return undefined;
        }

        return this._values[i];
    }

    set(key: string, value: T | undefined) {
        if (value === undefined) {
            this.delete(key);
            return;
        }

        const i = this.find(key);
        if (i < 0) {
            this._keys.push("" + key);
            this._values.push(value);
            return;
        }

        this._values[i] = value;
    }

    [key: string]: T | undefined;

    has(key: string): boolean {
        return this.find(key) >= 0;
    }

    delete(key: string): boolean {
        const i = this.find(key);
        if (i < 0) {
            return false;
        }

        this._keys.splice(i, 1);
        this._values.splice(i, 1);
        return true;
    }

    keys(): string[] {
        const r: string[] = [];
        for (let i = 0; i < this._keys.length; i++) {
            r.push("" + this._keys[i]);
        }

        return r;
    }
}

function objectDictionaryAssign<T>(target: ObjectDictionary<T>, source: ObjectDictionary<T>) {
    for (const k in source) {
        target[k] = source[k];
    }
}

class P {
    constructor(public n: string) {}
}

function make(m: string, i: number) {
    return { [m]: i, b: 2 };
}

function sum(d: { [key: string]: number }) {
    let total: number = 0;
    for (const k in d) {
        total += d[k] ?? 0;
    }

    return total;
}

// an interface of a string index signature and nothing else is the same dictionary
interface Counts {
    [name: string]: number;
}

interface Table<T> {
    [key: string]: T;
}

// ...and one extending it, adding nothing
interface MoreCounts extends Counts {}
interface Rows<T> extends Table<T> {}

function total(c: Counts) {
    let sum: number = 0;
    for (const k in c) sum += c[k] ?? 0;
    return sum;
}

function main() {
    const d: { [key: string]: number } = { a: 1, get: 5 };
    const k = "x" + 1;
    d[k] = 2;
    d.size = 9;
    assert(d["a"] == 1);
    assert(d[k] == 2);
    assert(d.get == 5);
    assert(d.size == 9);
    assert(d["zz"] === undefined);
    assert("a" in d);
    assert(!("zz" in d));
    assert(sum(d) == 17);

    delete d["get"];
    delete d.size;
    assert(!("get" in d));
    const keys = Object.keys(d);
    assert(keys.length == 2 && keys[0] == "a" && keys[1] == "x1");

    const e = make("m" + 1, 7);
    assert(e["m1"] == 7);
    assert(e.b == 2);

    const o: { [key: string]: P } = {};
    for (let i = 0; i < 100; i++) {
        const key = "p" + (i % 10);
        o[key] = new P("v" + i);
        if (i % 3 == 0) {
            o[key] = new P("w" + i);
        }

        if (i % 7 == 0) {
            delete o["p" + (i % 4)];
        }
    }

    const p = o["p9"];
    assert(p !== undefined && p.n == "w99");
    // the last delete, at i = 98, is of p2
    assert(Object.keys(o).length == 9);
    assert(!("p2" in o));

    const counts: Counts = { a: 1, b: 2 };
    counts["c"] = 3;
    assert(total(counts) == 6 && counts["z"] === undefined, "an interface of a string index signature");

    const names: Table<string> = {};
    const key = "k" + 1;
    names[key] = "v1";
    assert(names["k1"] == "v1" && names["k2"] === undefined, "a generic one");

    // a spread of a dictionary: of the same values (the library's objectDictionaryAssign), and of others, each value
    // cast (it was "the values' types have to be the same")
    const same: Counts = { ...counts, d: 4 };
    assert(total(same) == 10, "a spread of the same values");
    const ints: { [k: string]: int } = { p: 1, q: 2 };
    const widened: Counts = { ...ints, r: 3.5 };
    assert(widened["p"] == 1 && total(widened) == 6.5 && Object.keys(widened).length == 3, "a spread of other values");

    const more: MoreCounts = { m: 5 };
    const rows: Rows<boolean> = {};
    rows["r" + 1] = true;
    assert(total(more) == 5 && rows["r1"] === true, "an interface extending one");

    print("done.");
}

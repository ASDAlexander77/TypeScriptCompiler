// A string index signature, `{ [key: string]: T }`, is a dictionary: the default library's ObjectDictionary<T>,
// whose entries are known at run time. `d[k]`, `d["name"]` and `d.name` are its entries - never its members, so
// `d.get` and `d.size` are entries too - read as `T | undefined` (undefined for a key it does not hold) and
// written as T. An object literal with a key known only at run time (`{ [m]: i }`) makes one, and so does one
// given where a dictionary is wanted. `k in d`, `for...in`, `Object.keys` and `delete d[k]` go by its entries.
// Under -mm=own it owns its values: written in moves, read borrows, overwritten or deleted drops.
class P {
    constructor(public n: string) {}
}

function make(m: string, i: number) {
    return { [m]: i, b: 2 };
}

function sum(d: { [key: string]: number }) {
    let total = 0;
    for (const k in d) {
        total += d[k] ?? 0;
    }

    return total;
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
    assert(Object.keys(d).join(",") == "a,x1");

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
    assert(Object.keys(o).length == 10);

    print("done.");
}

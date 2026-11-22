// A dictionary whose values are a union holding strings (`{ [k: string]: number | string }`), read and stored
// somewhere else: into another dictionary, an array, a field, a local. Under -mm=own a read borrows the entry, and
// what is stored is a copy of it - as for a dictionary of strings. It was "a value that borrows the result of
// 'ObjectDictionary<...>.get' cannot be stored, returned or captured": the read is an optional of a tagged union,
// `(number | string) | undefined`, which was not copied (TSLANG 157). A key the dictionary does not hold reads
// as before in every model (a union holding nothing).
// The tests compile without the default library: its ObjectDictionary, as the compiler finds it by name, over two
// arrays.
@dictionary
class ObjectDictionary<T = any> {
    private _keys: string[] = [];
    private _values: T[] = [];

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

    delete(key: string): boolean {
        const i = this.find(key);
        if (i < 0) {
            return false;
        }

        this._keys.splice(i, 1);
        this._values.splice(i, 1);
        return true;
    }
}

class Box {
    v: number | string = 0;
}

function make(i: number) {
    const d: { [k: string]: number | string } = {};
    d["c"] = "x" + i;
    d["n"] = i + 0.5;
    return d;
}

function main() {
    const e: { [k: string]: number | string } = {};
    const arr: (number | string)[] = [];
    const bx = new Box();
    let l: number | string = 0;
    for (let i = 0; i < 1000; i++) {
        const d = make(i);
        e["z"] = d["c"];
        e["n"] = d["n"];
        arr.push(d["c"]);
        arr.push(d["n"]);
        bx.v = d["c"];
        l = d["c"];
        // what was read is dropped: the copies stay
        d["c"] = "over" + i;
        delete d["n"];
        // (a read compared with a number goes by a local: `e["n"] == 1.5` is refused under -mm=own yet)
        let n: number | string = 0;
        n = e["n"];
        assert(e["z"] == "x" + i && n == i + 0.5);
        assert(arr[i * 2] == "x" + i && arr[i * 2 + 1] == i + 0.5);
        assert(bx.v == "x" + i && l == "x" + i);
        assert(d["c"] == "over" + i && d["n"] === undefined);
    }

    // the dictionaries read from are gone
    l = e["n"];
    assert(e["z"] == "x999" && l == 999.5 && arr.length == 2000 && arr[0] == "x0" && arr[1999] == 999.5);
    l = e["z"];
    assert(bx.v == "x999" && l == "x999");

    // a number, and a key it does not hold
    const d2 = make(7);
    l = d2["n"];
    bx.v = d2["n"];
    assert(l == 7.5 && bx.v == 7.5);
    e["m"] = d2["missing"];
    assert(e["m"] === undefined);

    print("done.");
}

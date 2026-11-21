// A module importing another, with a generic class whose code boxes an object into `any` (as the library's
// Map does, throwing a RangeError): the keys adapter of what it boxed was made while the imported file was
// read as declarations - a declaration, made public - and the module failed: "'ts.Func' op symbol
// declaration cannot have public visibility". A dictionary keyed by an imported value is made here too.
import { prefix, count } from "./map_module";

class Failure {
    constructor(public message: string) {}
}

class Store<V> {
    private values: V[] = [];

    put(value: V) {
        if (this.values.length > 100) {
            throw new Failure("full");
        }

        this.values.push(value);
    }

    get length() {
        return this.values.length;
    }
}

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

    keys(): string[] {
        const r: string[] = [];
        for (let i = 0; i < this._keys.length; i++) {
            r.push("" + this._keys[i]);
        }

        return r;
    }
}

function main() {
    const s = new Store<boolean>();
    s.put(true);
    assert(s.length == 1);

    const d: { [key: string]: number } = {};
    for (let i = 0; i < count; i++) {
        d[prefix + i] = i;
    }

    const e = { [prefix]: count };
    assert(d["k2"] == 2);
    assert(e["k"] == 3);
    assert(Object.keys(d).length == 3);
    print("done.");
}

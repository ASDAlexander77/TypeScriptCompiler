// TSLANG 216: under -mm=own a call's drop facts name the argument whose object owns each place it may overwrite
// (`this._keys` of a dictionary's `set`, a bag's `this.names`): a borrow of something apart from that argument - an
// object this function made, given an array it did not - is not dropped by it (a dictionary written in a loop over
// an array was "'it' borrows an element but is used here after it may be released"); and a union of tuples cast to a
// member is a borrow of the union in every branch, which a function may return (`return <Long>x`).
// The tests compile without the default library: its ObjectDictionary, as the compiler finds it by name.
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

class Bag {
    private names: string[] = [];

    add(name: string) {
        this.names.push("" + name);
    }

    clear() {
        this.names.splice(0, this.names.length);
    }

    get size() {
        return this.names.length;
    }
}

// a dictionary made here, written in a loop over an array given
function countKeys(items: string[]) {
    const out: Record<string, number> = {};
    for (const it of items) {
        out[it] = 1;
    }

    return Object.keys(out).length;
}

// ...and an index signature, read as well as written
function countDistinct(items: string[]) {
    const out: { [key: string]: number } = {};
    for (const it of items) {
        out[it] = (out[it] ?? 0) + 1;
    }

    return Object.keys(out).length;
}

// a bag made here, emptied while an element of the array given is borrowed
function lastOf(items: string[]) {
    const bag = new Bag();
    for (const it of items) {
        bag.clear();
        bag.add(it);
    }

    return bag.size;
}

type Short = [number, string];
type Long = [number, string, number];

// a union of tuples cast to a member, returned: each branch borrows `x`
function widen(x: Short | Long) {
    return <Long>x;
}

function narrow(x: Short | Long) {
    return <Short>x;
}

function main() {
    let total = 0;
    for (let i = 0; i < 3; i++) {
        const items: string[] = [];
        items.push("k" + i);
        items.push("l" + i);
        items.push("k" + i);
        total += countKeys(items) + countDistinct(items) + lastOf(items);

        const x: Short | Long = [i, "a" + i];
        const l = widen(x);
        assert(l[1] == "a" + i && l[2] == 0);
        const y: Short | Long = [i, "b" + i, 7];
        assert(widen(y)[2] == 7 && narrow(y)[1] == "b" + i && narrow(x)[1] == "a" + i);
    }

    assert(total == 15);
    print("done.");
}

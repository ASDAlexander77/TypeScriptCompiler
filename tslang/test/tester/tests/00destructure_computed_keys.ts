// Object patterns: a computed key is the key read (it read the binding's name: `const { [itemId]: item } = items`
// was "Class member 'item' can't be found"), a dictionary's entries are read by key, and the elements after a nested
// pattern are declared too (TSLANG 191)

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

const itemId = "some-id";

function pick(n: number) {
    return "k" + n;
}

const byLiteral = ({ [`key`]: renamed }) => renamed;

function main() {
    const items: { [id: string]: number } = {};
    items[itemId] = 3;
    items["plain"] = 4;
    items[pick(1)] = 5;

    const { [itemId]: a } = items;
    assert(a == 3, "a constant computed key of a dictionary");

    const { plain } = items;
    assert(plain == 4, "a dictionary entry by its name");

    const { [pick(1)]: c } = items;
    assert(c == 5, "a key known at run time");

    const { missing = 9 } = items;
    assert(missing == 9, "the default of an absent entry");

    const o = { a: { x: 1 }, b: 2 };
    const { a: { x }, b } = o;
    assert(x == 1 && b == 2, "the element after a nested pattern");

    const k = "b";
    const { [k]: kb } = o;
    assert(kb == 2, "a constant computed key of an object");

    assert(byLiteral({ key: 7 }) == 7, "a literal computed key of a parameter pattern");

    print("done.");
}

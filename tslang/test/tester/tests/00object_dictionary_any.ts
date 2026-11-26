// not own: `any` has no keys under -mm=own
// Through `any`, a value's keys are looked up at run time, by the type it was boxed as: `delete x[k]` deletes a
// dictionary's entry, and is a TypeError for any other object, whose keys are fixed when it is made (ANY_KEYS_OP_DELETE);
// Object.isExtensible, isSealed and isFrozen ask how far it is closed (ANY_KEYS_OP_CLOSED) - a dictionary as far as
// Object.preventExtensions, seal and freeze closed it (ANY_KEYS_OP_CLOSE), another object sealed, an array open, a
// number frozen.
// The tests compile without the default library: its ObjectDictionary, as the compiler finds it by name.
@dictionary
class ObjectDictionary<T = any> {
    private _keys: string[] = [];
    private _values: T[] = [];
    private _closed: int = 0;

    close(level: int) {
        if (level > this._closed) {
            this._closed = level;
        }
    }

    closed(): int {
        return this._closed;
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
            if (this._closed > 0) {
                throw "not extensible";
            }

            this._keys.push("" + key);
            this._values.push(value);
            return;
        }

        if (this._closed == 3) {
            throw "frozen";
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

        if (this._closed >= 2) {
            throw "sealed";
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

class Point {
    constructor(public x: number) {}
}

function dictionary() {
    const d: { [key: string]: number } = { a: 1, b: 2 };
    const x: any = d;
    delete x["a"];
    delete x.none;
    assert(!("a" in d) && ("b" in d));

    assert(Object.isExtensible(x) && !Object.isSealed(x) && !Object.isFrozen(x));
    Object.seal(x);
    assert(Object.isSealed(x) && !Object.isFrozen(x) && !Object.isExtensible(d) && Object.isSealed(d));
    let threw = false;
    try {
        delete x["b"];
    } catch (e) {
        threw = true;
    }
    assert(threw && ("b" in d));

    Object.freeze(x);
    assert(Object.isFrozen(d));
}

function others() {
    const p: any = new Point(1);
    assert(!Object.isExtensible(p) && Object.isSealed(p) && !Object.isFrozen(p));
    Object.freeze(p);
    assert(!Object.isFrozen(p));

    let threw = false;
    try {
        delete p["x"];
    } catch (e) {
        threw = true;
    }
    assert(threw && p.x == 1);

    const a: any = [1, 2];
    assert(Object.isExtensible(a) && !Object.isSealed(a));

    const n: any = 5;
    const s: any = "s";
    assert(Object.isFrozen(n) && Object.isFrozen(s) && !Object.isExtensible(n));

    threw = false;
    try {
        delete s[0];
    } catch (e) {
        threw = true;
    }
    assert(threw);
}

function main() {
    dictionary();
    others();
    print("done.");
}

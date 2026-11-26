// Object's functions of a dictionary (`{ [key: string]: T }`), the one object that gains and loses keys: keys,
// getOwnPropertyNames, values and entries in the order its entries were added, hasOwn (`k in d`), assign (as a
// spread), fromEntries of [key, value] pairs. Object.preventExtensions, seal and freeze close it (its `close`) and
// Object.isExtensible, isSealed and isFrozen ask how far (`closed`). Of any other object, whose keys are fixed when
// it is made, isExtensible is false and isSealed true; an array grows, so it is extensible; a number is frozen.
// The tests compile without the default library: its ObjectDictionary and the functions the compiler calls by name.
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

function objectDictionaryAssign<T>(target: ObjectDictionary<T>, source: ObjectDictionary<T>) {
    for (const k in source) {
        target[k] = source[k];
    }
}

function objectDictionaryValues<T>(d: ObjectDictionary<T>): T[] {
    const r: T[] = [];
    for (const k in d) {
        r.push(<T>d[k]);
    }

    return r;
}

function objectDictionaryEntries<T>(d: ObjectDictionary<T>): [string, T][] {
    const r: [string, T][] = [];
    for (const k in d) {
        r.push([k, <T>d[k]]);
    }

    return r;
}

function objectDictionaryFromEntries<T, E>(entries: E): ObjectDictionary<T> {
    const d = new ObjectDictionary<T>();
    for (const e of entries) {
        d["" + e[0]] = <T>e[1];
    }

    return d;
}

function joined<T>(a: T[]) {
    let s = "";
    for (let i = 0; i < a.length; i++) {
        s += (i > 0 ? "," : "") + a[i];
    }

    return s;
}

function entries() {
    const d: { [key: string]: number } = { a: 1, b: 2 };
    d["c"] = 3;

    assert(joined(Object.keys(d)) == "a,b,c");
    assert(joined(Object.getOwnPropertyNames(d)) == "a,b,c");
    assert(joined(Object.values(d)) == "1,2,3");

    const es = Object.entries(d);
    assert(es.length == 3 && es[1][0] == "b" && es[1][1] == 2);

    assert(Object.hasOwn(d, "a") && !Object.hasOwn(d, "z"));

    const more: { [key: string]: number } = { c: 30, d: 4 };
    const same = Object.assign(d, more, { e: 5 });
    assert(joined(Object.keys(same)) == "a,b,c,d,e" && d["c"] == 30 && d["e"] == 5);

    const pairs: [string, number][] = [["x", 1], ["y", 2], ["x", 3]];
    const fromPairs = Object.fromEntries(pairs);
    assert(joined(Object.keys(fromPairs)) == "x,y" && fromPairs["x"] == 3);
}

function closing() {
    const d: { [key: string]: number } = { a: 1, b: 2 };
    assert(Object.isExtensible(d) && !Object.isSealed(d) && !Object.isFrozen(d));

    Object.preventExtensions(d);
    assert(!Object.isExtensible(d) && !Object.isSealed(d));
    let threw = false;
    try {
        d["c"] = 3;
    } catch (e) {
        threw = true;
    }
    assert(threw);
    delete d["b"];
    assert(!Object.hasOwn(d, "b"));

    assert(Object.seal(d) === d && Object.isSealed(d) && !Object.isFrozen(d));
    threw = false;
    try {
        delete d["a"];
    } catch (e) {
        threw = true;
    }
    assert(threw);
    d["a"] = 10;

    Object.freeze(d);
    threw = false;
    try {
        d["a"] = 2;
    } catch (e) {
        threw = true;
    }
    assert(Object.isFrozen(d) && threw && d["a"] == 10);

    Object.preventExtensions(d);
    assert(Object.isFrozen(d));
}

class Point {
    constructor(public x: number) {}

    get twice() {
        return this.x * 2;
    }

    norm() {
        return this.x;
    }
}

function others() {
    const p = new Point(1);
    assert(!Object.isExtensible(p) && Object.isSealed(p) && !Object.isFrozen(p));
    assert(Object.freeze(p) === p);
    assert(Object.hasOwn(p, "x") && !Object.hasOwn(p, "norm") && !Object.hasOwn(p, "twice"));

    const o = { a: 1 };
    assert(Object.hasOwn(o, "a") && !Object.hasOwn(o, "b") && Object.isSealed(o));

    const a = [1, 2];
    assert(Object.isExtensible(a) && !Object.isSealed(a));

    assert(Object.isFrozen(5) && Object.isFrozen("s"));
}

function main() {
    entries();
    closing();
    others();
    print("done.");
}

// TSLANG 215: `Record<string, T>` is `{ [key: string]: T }`, a dictionary of T (it was an object with no fields: the
// literal given to it lost its fields, and Object.keys gave none), both as the built-in Record and as a mapped type
// `{ [P in K]: T }` of K = string; a union of tuples cast to one of them (a shorter member's missing elements filled:
// USER RULING 2026-11-27), indexed (at a constant or a run-time index), iterated, and read as an array (`slice`,
// `forEach`, `[...x]`: of the member it holds; under -mm=own its strings copied).
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

type Rec<K extends string, T> = { [P in K]: T };

function keysOf(r: Record<string, number>) {
    return Object.keys(r).length;
}

function sumOf(r: Rec<string, number>) {
    let total: number = 0;
    for (const k in r) {
        total += r[k] ?? 0;
    }

    return total;
}

type Two = [number, string];
type Three = [number, string, string];

// a Record as a function's result type, read in another function (its members generated where it is read), and as a
// constraint, which an object of any fields meets
function recordOf(i: number): Record<string, number> {
    return { a: i, b: i + 1 };
}

function countOf<T extends Record<string, number>>(t: T) {
    let n = 0;
    for (const k in t) {
        n++;
    }

    return n;
}

function third(x: Two | Three) {
    return (<Three>x)[2];
}

function secondOf(x: Two | Three) {
    return x[1];
}

// a member of Array on a union of tuples is of the array of the member it holds (read as the first, the shorter one
// had three elements); under -mm=own a tuple's strings are copied for that array
function count(x: Two | Three) {
    let n = 0;
    x.forEach((e) => { n++; });
    return n;
}

function mapped(x: Two | Three) {
    let n = 0;
    for (const v of x.map((e) => 1)) {
        n += v;
    }

    return n;
}

function spreadLength(x: Two | Three) {
    return [...x].length;
}

function lengthOf(x: Two | Three) {
    return x.length;
}

function ofCount(x: Two | Three) {
    let n = 0;
    for (const e of x) {
        n++;
    }

    return n;
}

function at(x: Two | Three, i: number) {
    return x[i];
}

function arrays() {
    let total = 0;
    for (let i = 0; i < 3; i++) {
        const x: Two | Three = [i, "a" + i, "b" + i];
        const y: Two | Three = [i, "c" + i];
        total += count(x) + count(y);
        total += mapped(x) + mapped(y);
        total += spreadLength(x) + spreadLength(y);
        total += lengthOf(x) + lengthOf(y);
        total += ofCount(x) + ofCount(y);
        assert(at(x, 2) == "b" + i);
        assert(at(y, 1) == "c" + i);
        assert(at(x, 0) == i);
        const t: Two = [i, "d" + i];
        t.forEach((e) => { total++; });
    }

    assert(total == 81);
}

function tuples() {
    let total = 0;
    for (let i = 0; i < 3; i++) {
        const x: Two | Three = [i, "a" + i, "b" + i];
        const t = <Three>x;
        assert(t[2] == "b" + i);
        const u = x as [number, string, string];
        assert(u[1] == "a" + i);
        assert(x[1] == "a" + i);
        assert(x[0] == i);
        assert(third(x) == "b" + i);
        assert(secondOf(x) == "a" + i);
        total += x.length;

        const y: Two | Three = [i, "c" + i];
        assert(y[1] == "c" + i);
        assert(secondOf(y) == "c" + i);
        assert((<Two>y)[1] == "c" + i);
        total += y.length;

        // the value is no Three: the element it lacks is filled - "" for a string under strict null checks
        assert(third(y) == "");

        // ...0 for a number, false for a boolean, undefined where the type admits it (cast where it is read: under
        // -mm=own the cast is a merge of the members' branches, which a function can't return whole yet)
        const z: Two | Long = [i, "c" + i];
        const l = <Long>z;
        assert(l[0] == i && l[1] == "c" + i && l[2] == 0 && l[3] == false && l[4] == "" && l[5] === undefined);
        const w: Two | Long = [i, "e" + i, 7, true, "f" + i, 9];
        const m = <Long>w;
        assert(m[2] == 7 && m[3] && m[4] == "f" + i && m[5] == 9);
    }

    assert(total == 15);
}

type Long = [number, string, number, boolean, string, number | undefined];

function main() {
    tuples();
    arrays();

    const r: Record<string, number> = { a: 1, b: 2 };
    assert(Object.keys(r).length == 2);
    assert(r["a"] == 1);
    const k = "b";
    assert(r[k] == 2);
    assert(r["c"] === undefined);
    r["c"] = 3;
    assert(keysOf(r) == 3);

    const m: Rec<string, number> = { x: 10, y: 20 };
    assert(sumOf(m) == 30);

    let made = "";
    for (let i = 0; i < 3; i++) {
        const s: Record<string, string> = { first: "f" + i };
        s["second"] = "s" + i;
        made += s["first"] + s["second"];
    }

    assert(made == "f0s0f1s1f2s2");

    const fromCall = recordOf(4);
    assert(Object.keys(fromCall).length == 2 && fromCall["b"] == 5);
    assert(countOf({ x: 1, y: 2, z: 3 }) == 3);

    // a Record of known keys stays an object of those fields
    const p: Record<"a" | "b", number> = { a: 5, b: 6 };
    assert(p.a + p.b == 11);

    print("done.");
}

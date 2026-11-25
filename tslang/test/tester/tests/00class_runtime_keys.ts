// TSLANG 191: class members named by a key known only at run time (USER 2026-11-25: one table per class) - a method,
// a static value, a field declared with a type only. They are entries of the class's table, which a key read at run
// time (`h[k]`) finds after the class's own members. They were "a computed property name must be a constant".
// (the tests compile without the default library: its dictionary, as 00object_dictionary.ts declares it)
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

function pick(n: number): string {
    return "m" + n;
}

const methodName = pick(1);
const staticName = pick(2);
const declaredName = pick(3);
// a union of literals is a key known at run time too
const chosenName = methodName.length > 1 ? "u1" : "u2";

class Holder {
    n = 5;

    [methodName]() {
        return "value";
    }

    static [staticName] = 42;

    [declaredName]: string;

    [chosenName]() {
        return "chosen";
    }

    plain(): number {
        return this.n;
    }
}

function main() {
    const h = new Holder();
    assert(h.plain() == 5, "the class's own members");

    const k: string = methodName;
    const f = <() => string>h[k];
    assert(f() == "value", "a method named at run time");

    const s: string = staticName;
    assert(h[s] == 42, "a static value named at run time");

    const d: string = declaredName;
    assert(h[d] == undefined, "a field declared with a type only holds nothing");

    const u: string = chosenName;
    assert((<() => string>h[u])() == "chosen", "a method named by a union of literals");

    const n: string = "n";
    assert(h[n] == 5, "an own member by a run-time key");

    print("done.");
}

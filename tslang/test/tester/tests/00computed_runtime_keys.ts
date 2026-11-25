// TSLANG 191: an object literal with a key known only at run time is a dictionary - its methods too
// (`{ [name]() { ... } }`), and one given to an `any`. They were "a computed property name must be a constant" and
// "an object literal that makes a dictionary ... can hold only properties and spreads".
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
    return n > 0 ? "f" + n : "none";
}

function main() {
    const k1 = pick(1);
    const k2 = pick(2);

    const methods = {
        [k1]() {
            return 10;
        },
        [k2]() {
            return 20;
        },
    };
    assert(methods[k1]!() == 10, "a method under a run-time key");
    assert(methods["f2"]!() == 20, "read by its constant name");
    assert(methods["f3"] == undefined, "an absent key");

    const values = { a: 1, [k1]: 2 };
    assert(values[k1] == 2 && values.a == 1, "a property under a run-time key");

    const k3 = k1.length > 1 ? "u1" : "u2";
    const chosen = { [k3]: 3 };
    assert(chosen["u1"] == 3, "a key of a union of literals");

    const box: any = { [k2]: 7, b: 8 };
    assert(box[k2] == 7 && box.b == 8, "given to an any");

    print("done.");
}

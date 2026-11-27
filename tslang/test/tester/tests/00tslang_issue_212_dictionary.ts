// TSLANG 212 (152): a record's strings spread into a dictionary, or given to Object.assign - named or a literal - are
// copied under -mm=own (strings are values): the dictionary keeps what it is given ("a value that borrows a field cannot
// be stored"); the literal is given back after (it leaked). The tests compile without the default library: its
// ObjectDictionary, as 00object_dictionary_builtins.ts has it.
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

function main() {
    for (let i = 0; i < 3; i++) {
        const rec = { a: "x" + i, b: "y" };
        const d1: { [k: string]: string } = { ...rec };
        assert(d1["a"] == "x" + i && d1["b"] == "y", "spread");

        const d2: { [k: string]: string } = {};
        Object.assign(d2, rec);
        assert(d2["a"] == "x" + i && d2["b"] == "y", "Object.assign of a record");

        const d3: { [k: string]: string } = {};
        Object.assign(d3, { a: "z" + i, b: "w" });
        assert(d3["a"] == "z" + i && d3["b"] == "w", "Object.assign of a literal");
    }

    print("done.");
}

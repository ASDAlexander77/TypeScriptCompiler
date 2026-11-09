// A class whose indexer's `set` keeps the value it is given (issue 43): imported from a library,
// `bag[key] = value` hands the value to `set` as `bag.set(key, value)` does, so under -mm=own the
// caller doesn't free it too.
namespace M {
    export class Item {
        constructor(public n: number) {}
    }

    export class Bag {
        private keys: string[] = [];
        private values: Item[] = [];

        [key: string]: Item;

        @lends("this")
        get(key: string): Item {
            for (let i = 0; i < this.keys.length; i++) {
                if (this.keys[i] == key) {
                    return this.values[i];
                }
            }

            return this.values[0];
        }

        @lends("this")
        @keeps("value")
        set(key: string, value: Item): Bag {
            this.keys.push(key + "");
            this.values.push(value);
            return this;
        }

        get size() {
            return this.keys.length;
        }
    }
}

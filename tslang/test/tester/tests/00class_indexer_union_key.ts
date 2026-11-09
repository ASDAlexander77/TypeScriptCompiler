// A class indexer whose key is `number | string`: number keys and string keys, constant or not,
// go to the one get/set; a constant string that names a member is still that member.

class Table {
    private names: string[] = [];
    private items: string[] = [];

    [key: number | string]: string;

    get(key: number | string): string {
        let position = -1;
        if (typeof key == "number") {
            position = key;
        } else {
            for (let i = 0; i < this.names.length; i++) {
                if (this.names[i] == key) {
                    position = i;
                }
            }
        }

        if (position < 0 || position >= this.items.length) {
            return "";
        }

        return this.items[position];
    }

    set(key: number | string, value: string) {
        let position = -1;
        if (typeof key == "number") {
            position = key;
        } else {
            position = this.names.length;
            this.names.push(key + "");
            this.items.length = position + 1;
        }

        // in one place: under -mm=own a parameter is taken on every path or none
        this.items[position] = value;
    }

    get size() {
        return this.items.length;
    }
}

function main() {
    const table = new Table();

    // string keys
    table["a"] = "x";
    table["b"] = "z";
    assert(table["a"] == "x");
    assert(table["missing"] == "");

    // number keys reach the same items
    assert(table[0] == "x");
    assert(table[1] == "z");
    table[0] = "y";
    assert(table["a"] == "y");

    // variable keys of either kind
    const name = "b";
    let index = 1;
    let other = "a";
    assert(table[name] == "z");
    assert(table[index] == "z");
    assert(table[other] == "y");

    // a key that names a member is the member
    assert(table["size"] == 2);

    print(table["a"], table[1], table["size"]);
    print("done.");
}

// `obj["name"]` of a class with a string indexer: a key that names a member is that member,
// as in TypeScript; any other constant key goes to the indexer's get/set, as a variable key does.

class Base {
    label = "base";
}

class Bag extends Base {
    count = 0;

    private keys: string[] = [];
    private values: string[] = [];

    [key: string]: string;

    get(key: string): string {
        for (let i = 0; i < this.keys.length; i++) {
            if (this.keys[i] == key) {
                return this.values[i];
            }
        }

        return "";
    }

    set(key: string, value: string) {
        let position = -1;
        for (let i = 0; i < this.keys.length; i++) {
            if (this.keys[i] == key) {
                position = i;
            }
        }

        if (position < 0) {
            position = this.keys.length;
            this.keys.push(key + "");
            this.values.length = position + 1;
            this.count++;
        }

        // in one place: under -mm=own a parameter is taken on every path or none
        this.values[position] = value;
    }

    size() {
        return this.count;
    }
}

function main() {
    const bag = new Bag();

    // literal keys: the indexer
    bag["name"] = "Ann";
    bag["city"] = "Oslo";
    assert(bag["name"] == "Ann");
    assert(bag["city"] == "Oslo");
    assert(bag["missing"] == "");

    bag["name"] = "Bob";
    assert(bag["name"] == "Bob");

    // a const key is a constant too
    const key = "city";
    assert(bag[key] == "Oslo");

    // a variable key, as before
    let other = "name";
    assert(bag[other] == "Bob");

    // keys that name members still reach the members
    assert(bag["count"] == 2);
    assert(bag["label"] == "base");
    assert(bag["size"]() == 2);

    print(bag["name"], bag["city"], bag["count"]);
    print("done.");
}

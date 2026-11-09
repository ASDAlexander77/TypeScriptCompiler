// An indexer whose `set` is a method that returns something (TSLANG_ISSUES 43): `obj[key] = value`
// calls it and drops the result. The call was made with no result, unlike the function it calls,
// which crashed the compiler now and then while lowering to LLVM.
class Item {
    constructor(public n: number) {}
}

class Chain {
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

    // returns the object it is called on
    @keeps("value")
    set(key: string, value: Item): this {
        this.keys.push(key + "");
        this.values.push(value);
        return this;
    }

    get size() {
        return this.keys.length;
    }
}

class Counted {
    private values: number[] = [];

    [index: number]: number;

    get(index: number): number {
        return this.values[index];
    }

    // returns a value of its own
    set(index: number, value: number): number {
        while (this.values.length <= index) {
            this.values.push(0);
        }

        const previous = this.values[index];
        this.values[index] = value;
        return previous;
    }
}

function main() {
    const chain = new Chain();
    let a = "a";
    let b = "b";
    chain[a] = new Item(1);
    chain[b] = new Item(2);
    chain.set("c", new Item(3)).set("d", new Item(4));
    assert(chain.size == 4);
    assert(chain[a].n == 1);
    assert(chain[b].n == 2);
    assert(chain["d"].n == 4);

    const counted = new Counted();
    counted[2] = 5;
    counted[2] = 7;
    assert(counted[2] == 7);
    assert(counted[0] == 0);

    print("done.");
}

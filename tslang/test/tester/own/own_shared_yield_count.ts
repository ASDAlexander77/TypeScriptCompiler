// A generator over `this` yields a Shared<T> it reads into a state field: the yield hands the
// state's count to the caller, who owns what `next` returns. Leaving the loop early, or walking two
// entries, gives every count back (Map's values(), entries()).
class Node {
    constructor(public n: number) {}
}

type Entry = [key: string, value: Shared<Node>];

class Store {
    entries: Entry[] = [];

    put(key: string, value: Shared<Node>) {
        this.entries.push(<Entry>[key, value]);
    }

    @borrows("this")
    *values() {
        for (let i = 0; i < this.entries.length; i++) {
            const value = this.entries[i].value;
            yield value;
        }
    }

    @borrows("this")
    *pairs() {
        for (let i = 0; i < this.entries.length; i++) {
            const key = this.entries[i].key;
            const value = this.entries[i].value;
            yield <Entry>[key, value];
        }
    }
}

function walkValues(x: Shared<Node>, y: Shared<Node>) {
    const store = new Store();
    store.put("a", x);
    store.put("b", y);
    let total = 0;
    for (const v of store.values()) total += v.value.n;
    assert(total == 15);
}

function walkPairs(x: Shared<Node>, y: Shared<Node>) {
    const store = new Store();
    store.put("a", x);
    store.put("b", y);
    let total = 0;
    for (const [k, v] of store.pairs()) total += v.value.n;
    assert(total == 15);
}

function leaveEarly(x: Shared<Node>, y: Shared<Node>) {
    const store = new Store();
    store.put("a", x);
    store.put("b", y);
    for (const v of store.values()) {
        if (v.value.n == 7) break;
    }
}

function main() {
    const x = new Shared(new Node(7));
    const y = new Shared(new Node(8));
    walkValues(x, y);
    assert(Shared.count(x) == 1 && Shared.count(y) == 1);
    walkPairs(x, y);
    assert(Shared.count(x) == 1 && Shared.count(y) == 1);
    leaveEarly(x, y);
    assert(Shared.count(x) == 1 && Shared.count(y) == 1);
    print("done.");
}

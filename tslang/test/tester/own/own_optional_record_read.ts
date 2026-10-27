// A record found in an array is returned as `T | undefined`, a borrow of the array: printing it
// reads its fields out of the optional's tuple value, and a field that is a number keeps nothing.
// A caught value cast to its class is the exception's until the catch ends, so a call on it - a
// virtual one own has no facts for - cannot free it.
type Fruit = { name: string, quantity: int };

function find(this: Fruit[], test: (f: Fruit) => boolean): Fruit | undefined {
    for (const v of this) if (test(v)) return v;
    return undefined;
}

function made(n: int): Fruit[] {
    const r: Fruit[] = [];
    for (let i = 0; i < n; i++) {
        const f = { name: "f" + i, quantity: i };
        r.push(f);
    }

    return r;
}

class Failure {
    constructor(public code: int) {}

    isCode(c: int) {
        return this.code == c;
    }
}

function fail(c: int) {
    if (c > 0) throw new Failure(c);
}

function caughtCast(c: int) {
    let matched = false;
    try {
        fail(c);
    } catch (e) {
        const failure = <Failure>e;
        matched = failure.isCode(c) && failure instanceof Failure;
    }

    return matched;
}

function main() {
    const inventory = made(4);
    const found = inventory.find((f) => f.name == "f2");
    assert(found !== undefined && found.quantity == 2, "found");
    print(found);
    print(inventory.find((f) => f.quantity == 3));
    print(inventory.find((f) => f.quantity == 9));

    assert(caughtCast(5), "caught cast");
    assert(!caughtCast(0), "nothing thrown");

    print("done.");
}

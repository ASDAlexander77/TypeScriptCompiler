// The base method only reads its argument, the override keeps it: a virtual call may reach
// either, so the argument goes on the heap.
class Item {
    v: number;
    constructor(v: number) {
        this.v = v;
    }
}

class Visitor {
    visit(x: Item) {
        return x.v;
    }
}

class Keeper extends Visitor {
    visit(x: Item) {
        store.push(x);
        return 0;
    }
}

let store: Item[] = [];

function run(v: Visitor, i: number) {
    const item = new Item(i);
    return v.visit(item);
}

function main() {
    assert(run(new Visitor(), 5) == 5);
    run(new Keeper(), 6);
    assert(store.length == 1);
    assert(store[0].v == 6);
    print("done.");
}

// A method that keeps a closure capturing its parameter keeps the parameter (spec 76): the caller
// gives the argument up, so reading it after the call is a use after a move.
class Item {
    constructor(public n: number) {}
}

class Counter {
    get = () => 0;

    setup(item: Item) {
        this.get = () => item.n;
    }
}

function main() {
    const c = new Counter();
    const item = new Item(4);
    c.setup(item);
    print(item.n);
}

// A closure stored into a field of `this` that captures something besides `this` (a parameter) is no
// back-reference (spec 70): an escaping closure owns what it captures. The method keeps the parameter
// (spec 76), but `this` is the caller's, which no object it owns can own.
class Item {
    n: number = 2;
}

class Counter {
    n: number = 1;
    get = () => 0;

    setup(item: Item) {
        this.get = () => this.n + item.n;
    }
}

function main() {
    const c = new Counter();
    const item = new Item();
    c.setup(item);
    print(c.get());
}

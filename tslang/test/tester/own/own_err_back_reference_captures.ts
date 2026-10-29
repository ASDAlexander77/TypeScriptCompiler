// A closure stored into a field of `this` that captures something besides `this` (a parameter) is no
// back-reference (spec 70): an escaping closure owns what it captures, and the parameter is the caller's.
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

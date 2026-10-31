// -mm=own rejects: `inner` borrows a field of `last`, and a callback run across the call gives `last` a
// new value, freeing the field `inner` reads afterwards.
class Box {
    constructor(public n: number) {}
}

class Pair {
    inner: Box;
    constructor(n: number) { this.inner = new Box(n); }
}

class Bag {
    items: Box[] = [new Box(1), new Box(2)];

    forEach(f: (b: Box) => void) {
        for (const b of this.items) f(b);
    }
}

function main() {
    const bag = new Bag();
    let last = new Pair(0);
    const inner = last.inner;
    bag.forEach(b => { last = new Pair(b.n); });
    print(inner.n, last.inner.n);
}

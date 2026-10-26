// -mm=own: what a method calls through a callback parameter is what its callers pass (spec 33's
// callback families), for a method called through its class's vtable too, when no method overrides
// it. Here every callback frees nothing, so `forEach` may lend them its elements. A method no one
// calls with a callback (`each` below) calls nothing through it.
class Node {
    constructor(public n: number) {}
}

class Bag {
    items: Node[] = [];

    forEach(f: (s: Node) => void) {
        for (let i = 0; i < this.items.length; i++) f(this.items[i]);
    }

    each(f: (s: Node) => void) {
        for (let i = 0; i < this.items.length; i++) f(this.items[i]);
    }
}

// a class of the same shape, unrelated to Bag: its `forEach` overrides nothing of Bag's
class Other {
    items: Node[] = [];

    forEach(f: (s: Node) => void) {
        for (let i = 0; i < this.items.length; i++) f(this.items[i]);
    }
}

function show(s: Node) {
    print(s.n);
}

function main() {
    const b = new Bag();
    b.items.push(new Node(3));
    b.items.push(new Node(4));
    b.forEach((s) => print(s.n));
    b.forEach(show);

    const o = new Other();
    o.items.push(new Node(5));
    o.forEach(show);

    print("done.");
}

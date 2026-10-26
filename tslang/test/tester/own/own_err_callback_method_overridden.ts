// -mm=own rejects: `forEach` lends an element to its callback, and a call through Bag's vtable may
// reach Bag2's override, so the callbacks Bag's callers pass are not all the method may call through
// `f` - nothing says it frees nothing. (Without the override, the callbacks' facts are its.)
class Node {
    constructor(public n: number) {}
}

class Bag {
    items: Node[] = [];

    forEach(f: (s: Node) => void) {
        for (let i = 0; i < this.items.length; i++) f(this.items[i]);
    }
}

class Bag2 extends Bag {
    forEach(f: (s: Node) => void) {
        this.items = [];
    }
}

function main() {
    const b: Bag = new Bag2();
    b.items.push(new Node(3));
    b.forEach((s) => print(s.n));
}

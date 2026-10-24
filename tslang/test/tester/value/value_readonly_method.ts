// The fields of a @value value are readonly: a method does not assign them.
class Node {
    constructor(public n: int) {}
}

@value
class H {
    constructor(public readonly node: Node) {}

    set(n: int) {
        this.node = new Node(n);
        return this.node.n;
    }
}

function main() {
    const h = new H(new Node(1));
    print(h.set(5));
}

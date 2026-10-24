// The fields of a @value value are readonly: only its own constructor's body assigns them, through
// `this`, and a closure inside the constructor is not that body.
class Node {
    constructor(public n: int) {}
}

@value
class H {
    readonly node: Node;

    constructor(n: int) {
        this.node = new Node(0);
        const f = () => {
            this.node = new Node(n);
        };
        f();
    }
}

function main() {
    const h = new H(7);
    print(h.node.n);
}

// A closure in a @value constructor that reads `this` holds the address of the slot `new` made,
// which the closure's box borrows, as does a closure inside it that reads `this` through the outer
// box: under rc a box's release routine frees the box alone, and no function takes or gives back a
// count of the slot as if it were a cell (value-ctor-capture.cmake). A closure in a method takes a
// copy of `this` (the value), not a cell, so its box frees itself alone too.
@value
class W {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        const f = () => this.a + 1;
        this.b = f();
    }
}

@value
class W2 {
    readonly a: number;
    readonly b: number;

    constructor(a: number) {
        this.a = a;
        const f = () => {
            const h = () => this.a + 1;
            return h();
        };
        this.b = f();
    }
}

class Node {
    constructor(public n: int) {}
}

@value
class M {
    constructor(public readonly node: Node) {}

    reader() {
        return () => this.node.n;
    }
}

function main() {
    assert(new M(new Node(3)).reader()() == 3);
    assert(new W(4).b == 5);
    assert(new W2(4).b == 5);
    print("done.");
}

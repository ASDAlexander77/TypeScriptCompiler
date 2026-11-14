// TSLANG_ISSUES 72: a local kept by a constructor reads the object's field after, so the object popped
// before the read frees it.
class Node2 {
    constructor(public kind: int, public t: number[]) {}
}

class App {
    nodes: Node2[] = [];
    first = 0.0;

    build(o: int) {
        const t: number[] = [1.0];
        const node = new Node2(o, t);
        this.nodes.push(node);
        this.nodes.pop();
        this.first = t[0];
    }
}

function main() {
    new App().build(1);
    print("done.");
}

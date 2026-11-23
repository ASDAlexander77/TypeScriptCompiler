// -mm=own --own-fallback=count: a new node, held by a const, stored in two holders. Own refuses
// the second store (the const borrows 'h.node' after the first); with the flag the node is counted.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        const n = new Node(i, "a" + i);
        h.node = n;
        g.node = n;
        sum += h.node.v + g.node.v;
        if (h.node.s != g.node.s) print("bad");
    }

    print(sum);
    print("done.");
}

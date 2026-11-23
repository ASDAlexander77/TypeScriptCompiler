// -mm=own --own-fallback=count: a node held by a field, given twice to one call that keeps both of
// those parameters. With the flag the call gets two retains of its own, one per kept parameter
// (one retain for two shares would free the node while a holder still has it).
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function keep2(h: Holder, g: Holder, a: Node, b: Node) {
    h.node = a;
    g.node = b;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        const x = new Holder();
        const n = new Node(i, "a" + i);
        x.node = n;
        keep2(h, g, n, n);
        sum += h.node!.v + g.node!.v + x.node.v;
    }

    print(sum);
    print("done.");
}

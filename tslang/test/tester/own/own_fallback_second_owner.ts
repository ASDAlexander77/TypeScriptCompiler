// -mm=own --own-fallback=count: a node stored in two holders by a function that keeps it. Own
// refuses the second store ('n' is used after its value was moved); with the flag the node is
// counted, and each holder's release gives back one owner. Both holders read it after.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function link(h: Holder, g: Holder, n: Node) {
    h.node = n;
    g.node = n;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        link(h, g, new Node(i, "a" + i));
        sum += h.node!.v + g.node!.v;
        if (h.node!.s != g.node!.s) print("bad");
    }

    print(sum);
    print("done.");
}

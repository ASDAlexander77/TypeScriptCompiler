// -mm=own --own-fallback=count: one new node given to two calls that keep it. Own refuses the second
// call ('n' is used after its value was moved); rc gives a keeping call no retain, so with the flag
// each call gets a retain of its own and the node's release stays.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function keep1(h: Holder, n: Node) {
    h.node = n;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        const n = new Node(i, "a" + i);
        keep1(h, n);
        keep1(g, n);
        sum += h.node!.v + g.node!.v;
        if (h.node!.s != g.node!.s) print("bad");
    }

    print(sum);
    print("done.");
}

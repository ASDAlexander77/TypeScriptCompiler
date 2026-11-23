// -mm=own --own-fallback=count: the second-owner shape in a function that throws after the second
// store, caught in main. The holders keep the node; what unwinding releases must leave it whole.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function mk(i: number) {
    return new Node(i, "a" + i);
}

function link(h: Holder, g: Holder, i: number) {
    const n = mk(i);
    h.node = n;
    g.node = n;
    if (i >= 0) throw "stop";
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        try {
            link(h, g, i);
        } catch (e) {
            sum += h.node!.v + g.node!.v;
        }
    }

    print(sum);
    print("done.");
}

// -mm=own --own-fallback=count still rejects: `g.node = f(h)` where `f` returns `h.node`. The caller
// has no retain for the result (rc takes its count inside `f`, which own erases because `f` lends),
// so there is nothing to keep: a borrow of a call's result stays an error with the flag (tier 2).
class Node {
    constructor(public v: number) {}
}

class Holder {
    node: Node | null = null;
}

function f(h: Holder) {
    return h.node!;
}

function main() {
    const h = new Holder();
    const g = new Holder();
    h.node = new Node(1);
    g.node = f(h);
    print(g.node.v);
    print("done.");
}

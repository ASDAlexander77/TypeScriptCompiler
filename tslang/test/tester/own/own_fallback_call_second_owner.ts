// -mm=own --own-fallback=count: a call's result, held by a const, stored in two holders. The result
// arrives owning its count; with the flag it is counted, and every retain rc placed for it stays.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function mk(i: number) {
    return new Node(i, "a" + i);
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        const n = mk(i);
        h.node = n;
        g.node = n;
        sum += h.node.v + g.node.v;
        if (h.node.s != g.node.s) print("bad");
    }

    print(sum);
    print("done.");
}

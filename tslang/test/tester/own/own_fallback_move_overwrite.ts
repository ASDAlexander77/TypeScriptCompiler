// -mm=own --own-fallback=count: `h.node = n; h.node = other; g.node = n`. The overwrite releases
// the holder's share of `n` before `g.node` takes one: keeping only the later retain would free
// `n` at the overwrite. Every retain and release rc placed for `n` stays.
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
        h.node = mk(i + 1);
        g.node = n;
        sum += g.node.v + h.node.v;
        if (g.node.s != "a" + i) print("bad");
    }

    print(sum);
    print("done.");
}

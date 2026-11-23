// -mm=own --own-fallback=count: a counted object captured by a closure kept in a module-level keeper
// (an append-only array a global holds). `link` stores the node in two holders, which own refuses
// (a second owner); with the flag the node is counted. `remember` keeps a closure for good, and the
// node it captures is made immortal there (ts.MakeImmortal) while the node has an extra owner: that
// discards its counts, which is safe - nothing frees an immortal block - so both holders let go of
// it and the closure still reads it.
class Node {
    constructor(public v: number) {}
}

class Holder {
    node: Node | null = null;
}

const handlers: (() => number)[] = [];

function remember(n: Node) {
    handlers.push(() => n.v);
}

function link(h: Holder, g: Holder, n: Node) {
    h.node = n;
    g.node = n;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 100; i++) {
        const h = new Holder();
        const g = new Holder();
        link(h, g, new Node(i));
        remember(h.node!);
        h.node = null;
        sum += g.node!.v;
        g.node = null;
    }

    for (const f of handlers) sum += f();
    print(sum);
    print(handlers.length);
    print("done.");
}

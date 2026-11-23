// -mm=own --own-fallback=count: a WeakRef to a counted object. `link` stores the child in two
// holders, which own refuses (a second owner); with the flag the child is counted. The first holder
// to let go leaves it alive, the last one destroys it - a tombstone while the WeakRef remains, freed
// with the WeakRef - and deref() then gives undefined. Its @weak parent field reads the parent while
// both live. A loop, so that a leaked child or tombstone shows as growth (mm_fuzz.py leaks).
class Node {
    @weak parent: Node | null = null;
    constructor(public v: number) {}
}

class Holder {
    node: Node | null = null;
}

function link(h: Holder, g: Holder, n: Node) {
    h.node = n;
    g.node = n;
}

// under gc an object nobody holds may still be found
function freesAtOnce() {
    return Shared.count(new Shared(0)) >= 0;
}

function main() {
    const own = freesAtOnce();
    const root = new Node(0);
    let alive = 0;
    let gone = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        const child = new Node(i);
        child.parent = root;
        link(h, g, child);
        const w = new WeakRef(h.node!);
        if (w.deref()) alive++;
        h.node = null;
        if (!w.deref()) print("freed with an owner left");
        if (g.node!.parent!.v != 0) print("bad parent");
        g.node = null;
        if (!own || !w.deref()) gone++;
    }

    print(alive == 1000 ? "alive" : "not alive");
    print(gone == 1000 ? "gone" : "still there");
    print("done.");
}

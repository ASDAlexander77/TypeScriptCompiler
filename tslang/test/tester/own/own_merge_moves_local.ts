// An owned local chosen by `?:` against null or undefined, and returned or stored: it moves on the way
// that chooses it, and is released on the way that does not (`return set ? n : undefined`).
class Node {
    name = "";
}

class Holder {
    n: Node | null = null;
}

function named(set: boolean, name: string): Node | undefined {
    const n = new Node();
    n.name = name;
    return set ? n : undefined;
}

function orNull(set: boolean, name: string): Node | null {
    const n = new Node();
    n.name = name;
    return set ? n : null;
}

function store(h: Holder, set: boolean, name: string) {
    const n = new Node();
    n.name = name;
    h.n = set ? n : null;
}

function main() {
    const h = new Holder();
    for (let i = 0; i < 200; i++) {
        const a = named(i % 2 == 0, "a" + i);
        assert(i % 2 == 0 ? a?.name == "a" + i : a === undefined, "named");
        const b = orNull(i % 3 == 0, "b" + i);
        assert(i % 3 == 0 ? b?.name == "b" + i : b === null, "orNull");
        store(h, i % 2 == 1, "h" + i);
        assert(i % 2 == 1 ? h.n?.name == "h" + i : h.n === null, "store");
    }

    print("done.");
}

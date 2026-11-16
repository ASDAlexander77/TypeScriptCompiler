// -mm=own: `?:` over two fresh locals (`return set ? n : m`, a select): each moves into what is
// selected, and the one not selected is released right there. Names are made at run time, so a
// double release or a lost one shows.
class Node {
    name = "";
    items: number[] = [];
}

function churn() {
    const junk: Node[] = [];
    for (let i = 0; i < 64; i++) {
        const j = new Node();
        j.items.push(i);
        junk.push(j);
    }
}

function pick(set: boolean, i: number): Node {
    const n = new Node();
    n.name = "n" + i;
    n.items.push(i);
    const m = new Node();
    m.name = "m" + i;
    return set ? n : m;
}

function intoLocal(set: boolean, i: number) {
    const n = new Node();
    n.name = "a" + i;
    const m = new Node();
    m.name = "b" + i;
    const chosen = set ? n : m;
    churn();
    return chosen.name;
}

class Holder {
    node: Node = new Node();
}

function store(h: Holder, set: boolean, i: number) {
    const n = new Node();
    n.name = "x" + i;
    const m = new Node();
    m.name = "y" + i;
    h.node = set ? n : m;
}

function main() {
    const h = new Holder();
    for (let i = 0; i < 2000; i++) {
        const k = i % 10;
        const p = pick(k % 2 == 0, k);
        churn();
        assert(k % 2 == 0 ? p.name == "n" + k && p.items[0] == k : p.name == "m" + k, "pick");

        assert(intoLocal(k % 3 == 0, k) == (k % 3 == 0 ? "a" + k : "b" + k), "into a local");

        store(h, k % 2 == 1, k);
        churn();
        assert(h.node.name == (k % 2 == 1 ? "x" + k : "y" + k), "store");
    }

    print("done.");
}

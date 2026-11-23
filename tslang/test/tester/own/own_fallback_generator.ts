// -mm=own --own-fallback=count: a generator stores one node in two holders across a `yield`, and is
// iterated half-way and dropped. The node is counted; dropping the generator gives back its owners.
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function mk(i: number) {
    return new Node(i, "a" + i);
}

function* pairs(n: int) {
    for (let i = 0; i < n; i++) {
        const h = new Holder();
        const g = new Holder();
        const x = mk(i);
        h.node = x;
        g.node = x;
        yield h.node.v;
        yield g.node.v;
    }
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        let k = 0;
        for (const v of pairs(4)) {
            sum += v;
            if (++k >= 3) break;
        }
    }

    print(sum);
    print("done.");
}

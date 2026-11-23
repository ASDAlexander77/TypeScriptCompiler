// -mm=own --own-fallback=count: a field of a global's object given to a call that keeps it and
// replaces the global. Own refuses the borrow given away; with the flag the node is counted, and the
// call is given one count of its own (one retain before it, none after: the callee keeps it).
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

let current = new Holder();

function keep(n: Node, into: Holder) {
    current = new Holder();
    into.node = n;
    return n.v;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        current.node = new Node(i, "a" + i);
        const g = new Holder();
        sum += keep(current.node!, g);
        if (g.node!.s != "a" + i) print("bad");
    }

    print(sum);
    print("done.");
}

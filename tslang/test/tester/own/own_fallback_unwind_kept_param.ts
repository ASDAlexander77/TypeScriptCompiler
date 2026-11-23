// -mm=own --own-fallback=count: a counted parameter the function keeps twice, and then a call that
// throws. The parameter's share (the caller moved it in) is given back at the function's exits; an
// exception leaving through the call must give it back too, or the node outlives its holders (a
// leak mm_fuzz.py leaks shows).
class Node {
    constructor(public v: number, public s: string) {}
}

class Holder {
    node: Node | null = null;
}

function boom(i: number) {
    if (i >= 0) throw "stop";
}

function link(h: Holder, g: Holder, n: Node, i: number) {
    h.node = n;
    g.node = n;
    boom(i);
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = new Holder();
        const g = new Holder();
        try {
            link(h, g, new Node(i, "a" + i), i);
        } catch (e) {
            sum += h.node!.v + g.node!.v;
        }
    }

    print(sum);
    print("done.");
}

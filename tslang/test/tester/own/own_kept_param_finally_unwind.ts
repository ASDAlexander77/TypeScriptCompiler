// -mm=own: a parameter the function keeps on some paths (`if (c % 2 == 0) h.node = n`), with the exception leaving
// through the function's own `finally`: the parameter is released at the function's exits on the paths that did not
// keep it, and the end of that cleanup - where the exception leaves the function - was not one of them: it leaked
// (+9.9 MB per 400 runs, final-review M-1). Kept and then thrown past (`mayThrow` after the store) it stays the
// holder's. The same program under gc and rc.
class Node {
    constructor(public v: number, public s: string) {}
}
class Holder {
    node: Node | null = null;
}
let trace = 0;
function keep(h: Holder, n: Node, c: number): number {
    try {
        if (c % 7 == 0) throw "x";
        if (c % 2 == 0) h.node = n;
        return 1;
    } finally {
        trace++;
    }
}
@noinline function mayThrow(c: number) { if (c % 5 == 0) throw "y"; }
function keep2(h: Holder, n: Node, c: number): number {
    try {
        if (c % 2 == 0) h.node = n;
        mayThrow(c);
        if (c % 7 == 0) throw "x";
        return 1;
    } finally {
        trace++;
    }
}
function main() {
    let sum: number = 0;
    trace = 0;
    for (let i = 0; i < 4000; i++) {
        const h = new Holder();
        try {
            sum += keep(h, new Node(i, "a" + i), i);
        } catch (e) {
            sum += 3;
        }
        const h2 = new Holder();
        try {
            sum += keep2(h2, new Node(i, "b" + i), i);
        } catch (e) {
            sum += 3;
        }
        if (h.node) { sum += h.node.v; if (h.node.s != "a" + i) print("bad h"); }
        if (h2.node) { sum += h2.node.v; if (h2.node.s != "b" + i) print("bad h2"); }
    }
    print(sum + trace);
    assert(sum + trace == 7445088, "total");
    print("done.");
}


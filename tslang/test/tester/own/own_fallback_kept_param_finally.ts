// -mm=own --own-fallback=count: a counted parameter the function keeps, with the exception leaving through the
// function's own `finally` (`try { h.node = n; if (c) throw "x"; return 1 } finally { g.node = n }`): its share was
// given back at the returns and the throws, not at the end of that cleanup, and leaked (+9.9 MB per 400 runs; plain
// own the same, own_kept_param_finally_unwind). Prints what gc prints.
class Node {
    constructor(public v: number, public s: string) {}
}
class Holder {
    node: Node | null = null;
}
function link(h: Holder, g: Holder, n: Node, c: number): number {
    try {
        h.node = n;
        if (c % 7 == 0) throw "x";
        return 1;
    } finally {
        g.node = n;
    }
}
function main() {
    let sum: number = 0;
    for (let i = 0; i < 4000; i++) {
        const h = new Holder();
        const g = new Holder();
        try {
            sum += link(h, g, new Node(i, "a" + i), i);
        } catch (e) {
            sum += 3;
        }
        if (h.node) { sum += h.node.v; if (h.node.s != "a" + i) print("bad h"); }
        if (g.node) { sum += g.node.v; if (g.node.s != "a" + i) print("bad g"); }
    }
    print(sum);
    print("done.");
}

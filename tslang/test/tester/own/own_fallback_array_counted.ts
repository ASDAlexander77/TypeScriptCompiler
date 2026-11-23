// -mm=own --own-fallback=count: a counted object (a holder's field and an array both own it) that
// the array lets go of by pop, splice, a shorter length and its own release, in both store orders.
// The array's release takes one of its owners; the one left keeps it, and the last one frees it.
class Node {
    constructor(public v: number, public s: string) {}
}
class Holder {
    node: Node | null = null;
}
function linkFieldFirst(h: Holder, a: Node[], n: Node) {
    h.node = n;
    a.push(n);
}
function linkArrayFirst(h: Holder, a: Node[], n: Node) {
    a.push(n);
    h.node = n;
}
function main() {
    let sum: number = 0;
    for (let i = 0; i < 3000; i++) {
        // the array lets go first: the holder's field keeps the object
        const g = new Holder();
        const a: Node[] = [];
        linkFieldFirst(g, a, new Node(i, "a" + i));
        if (i % 4 == 0) a.pop();
        else if (i % 4 == 1) a.splice(0, 1);
        else if (i % 4 == 2) a.length = 0;
        const junk = new Node(-1, "zzzzzzzz" + i);
        sum += g.node!.v + junk.v;
        if (g.node!.s != "a" + i) print("bad");
    }
    for (let i = 0; i < 3000; i++) {
        // the field lets go first: the array keeps the object, then pops it
        const a: Node[] = [];
        const g = new Holder();
        linkArrayFirst(g, a, new Node(i, "b" + i));
        g.node = null;
        const junk = new Node(-1, "yyyyyyyy" + i);
        sum += a[0].v + junk.v;
        if (a[0].s != "b" + i) print("bad");
        if (i % 2 == 0) { const p = a.pop()!; sum += p.v; }
    }
    print(sum);
    print("done.");
}

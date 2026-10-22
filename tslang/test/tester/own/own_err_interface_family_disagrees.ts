// -mm=own rejects: two implementations of a closed interface that disagree - one returns a borrow of
// its argument, the other a value of its own. A call through the interface cannot be told which, so
// it is given neither, and the one that borrows has to say so to its callers (spec 31).
class V {
    constructor(public x: number) {}
}
interface Getter { get(v: V): V; }
class GA implements Getter { get(v: V): V { return v; } }
class GB implements Getter { get(v: V): V { return new V(v.x + 1); } }
function main() {
    const gs: Getter[] = [new GA(), new GB()];
    const a = new V(5);
    let s: number = 0;
    for (const g of gs) { const b = g.get(a); s += b.x; }
    print(s, a.x);
}

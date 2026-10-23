// -mm=own rejects: a `Th` value made out of a `J` value reaches `J`'s implementations through `Th`'s
// slots, which this module does not tie to `Th`. Both interfaces are open, and `G.get`, which returns
// a borrow of its argument, cannot tell the callers through either (spec 31).
class V {
    constructor(public x: number) {}
}
interface Th { get(v: V): V; }
interface J extends Th { n(): number; }
class G implements J { n(): number { return 1; } get(v: V): V { return v; } }
function via(t: Th, a: V) { const b = t.get(a); return b.x; }
function main() {
    const j: J = new G();
    const a = new V(5);
    print(via(j, a), a.x);
}

// -mm=own rejects: the plan's closed-interface program (own_interface_facts) with one implementation
// exported. Another module may extend `B` or implement `Th` and hand this module the value, so a call
// through `Th` may reach a function this module never sees: it gets no facts, may destroy anything,
// and the borrow of `sc.ts` it is made through ends at it.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
interface Th { area(v: V): number; }
class A implements Th { area(v: V) { return v.x; } }
export class B implements Th { area(v: V) { return v.x * 2; } }
class Sc { ts: Th[] = [new A(), new B()]; }
function tot(sc: Sc, r: V) {
    let best: number = 0;
    for (let i in sc.ts) { let a = sc.ts[i].area(r); if (a > best) best = a; }
    return best + r.x;
}
function main() { print(tot(new Sc(), new V(3))); print("done."); }

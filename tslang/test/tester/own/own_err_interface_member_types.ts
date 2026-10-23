// -mm=own rejects: an interface value made out of another one whose members take different
// interfaces of the same shape: a `HasF2` (`f: (o: Op2, ...)`) seen as a `HasF` (`f: (o: Op, ...)`)
// hands an `Op` to `f2`, whose call through `Op2` then reaches `OB.run`, which overwrites `h.c`.
// Every interface the two sides' members mention is open (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
interface Op { run(h: H): number; }
interface Op2 { run(h: H): number; }
class OA2 implements Op2 { run(h: H): number { return h.c.x; } }
class OB implements Op { run(h: H): number { h.c = new V(9); return 0; } }
function f2(o: Op2, h: H) { const c = h.c; o.run(h); churn(); return c.x; }
interface HasF2 { f: (o: Op2, h: H) => number; }
interface HasF { f: (o: Op, h: H) => number; }
function main() { const x: HasF2 = { f: f2 }; print(x.f(new OA2(), new H())); const y: HasF = x; print(y.f(new OB(), new H())); }

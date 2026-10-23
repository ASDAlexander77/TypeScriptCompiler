// -mm=own rejects: v11f's shape whose `Runner2` values are made only from a class object (the first
// call goes to the class directly): a `Runner2` seen as a `Runner` hands an `Op` to `R.go`, whose call
// through `Op2` reaches `OB.run`, which overwrites `h.c`. The class's vtable for `Runner2` gives its
// members' types (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
interface Op { run(h: H): number; }
interface Op2 { run(h: H): number; }
class OA2 implements Op2 { run(h: H): number { return h.c.x; } }
class OB implements Op { run(h: H): number { h.c = new V(9); return 0; } }
function f2(o: Op2, h: H) { const c = h.c; o.run(h); churn(); return c.x; }
interface Runner { go(o: Op, h: H): number; }
interface Runner2 { go(o: Op2, h: H): number; }
class R implements Runner2 { go(o: Op2, h: H) { const c = h.c; o.run(h); churn(); return c.x; } }
function call(r2: Runner2) { const r: Runner = r2; return r.go(new OB(), new H()); }
function main() { const rr = new R(); print(rr.go(new OA2(), new H())); const r2: Runner2 = rr; print(call(r2)); }

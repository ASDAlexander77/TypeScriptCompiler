// -mm=own rejects: a function taking an `Op2` reinterpreted through `any` as one taking an `Op`: the
// call through `Op2` inside it may reach `OB.run`, which overwrites `h.c`, so it may destroy what `c`
// borrows (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
interface Op { run(h: H): number; }
interface Op2 { run(h: H): number; }
class OA2 implements Op2 { run(h: H): number { return h.c.x; } }
class OB implements Op { run(h: H): number { h.c = new V(9); return 0; } }
function f2(o: Op2, h: H) { const c = h.c; o.run(h); churn(); return c.x; }
function main() {
  const h = new H();
  const fn = f2 as any as ((o: Op, h: H) => number);
  print(fn(new OB(), h));
  print(f2(new OA2(), h));
}

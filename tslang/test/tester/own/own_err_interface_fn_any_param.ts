// -mm=own rejects: a function taking an `Op2` reinterpreted through `any` as one taking an `Op`,
// so an `Op` value reaches `Op2`'s calls unseen. Both interfaces are open, and `G.take`, which returns a
// borrow of its argument, cannot tell the callers through either (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
interface Op { take(v: V): V; }
interface Op2 { get(v: V): V; }
class G implements Op { take(v: V): V { return v; } }
class G2 implements Op2 { get(v: V): V { return new V(v.x + 1); } }
function f2(o: Op2, a: V) { const b = o.get(a); return b.x; }
function main() {
  const a = new V(5);
  const fn = f2 as any as ((o: Op, a: V) => number);
  print(fn(new G(), a));
  churn();
  print(a.x, f2(new G2(), a));
}

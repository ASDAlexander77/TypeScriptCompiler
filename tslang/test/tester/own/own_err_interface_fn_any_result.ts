// -mm=own rejects: a function returning an `Op` reinterpreted through `any` as one returning an
// `Op2`, so an `Op` value reaches `Op2`'s calls unseen. Both interfaces are open, and `G.take`, which
// returns a borrow of its argument, cannot tell the callers through either (spec 31).
class V { constructor(public x: number) {} }
class H { c: V = new V(5); }
function churn() { let keep: V[] = []; for (let i = 0; i < 2000; i++) { keep.push(new V(77777)); } return keep.length; }
interface Op { take(v: V): V; }
interface Op2 { get(v: V): V; }
class G implements Op { take(v: V): V { return v; } }
class G2 implements Op2 { get(v: V): V { return new V(v.x + 1); } }
function main() {
  const a = new V(5);
  const o2: Op2 = new G2();
  { const b0 = o2.get(a); print(b0.x); }
  const mk = (): Op => new G();
  const mk2 = mk as any as (() => Op2);
  const o = mk2();
  { const b = o.get(a); print(b.x); }
  churn();
  print(a.x);
}

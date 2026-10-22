class W { constructor(public x: number) {} }
class V { w: W; constructor(public x: number) { this.w = new W(x); } }
class N { items: V[]; constructor(public v: V) { this.items = [new V(v.x)]; } }
class H { n: N = new N(new V(0)); }
@noinline
function mk(i: number) { return new N(new V(i)); }
function churn() { const hold: V[] = []; for (let i = 0; i < 64; i++) hold.push(new V(777)); }
@noinline
function f(p: N, q: () => number, z: number) { let r = p; r = new N(new V(2)); churn(); return q() + r.v.x - 2 + z; }
function main() { let s: number = 0; for (let i = 0; i < 1000; i++) { let n = mk(i); const g = () => n.v.x; s += f(n, g, (n = mk(5), 0)) + 2; } print(s); }

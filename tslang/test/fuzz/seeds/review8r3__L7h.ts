class W { constructor(public x: number) {} }
class V { w: W; constructor(public x: number) { this.w = new W(x); } }
class N { items: V[]; constructor(public v: V) { this.items = [new V(v.x)]; } }
class H { n: N = new N(new V(0)); }
@noinline
function mk(i: number) { return new N(new V(i)); }
function churn() { const hold: V[] = []; for (let i = 0; i < 64; i++) hold.push(new V(777)); }
class H2 { n: N = new N(new V(0)); }
@noinline
function f(h: H2, p: N, z: number) { h.n = p; churn(); return h.n.v.x + 2 + z; }
function main() { const h = new H2(); let s: number = 0; for (let i = 0; i < 1000; i++) { let n = mk(i); s += f(h, n, (n = mk(7), 0)); } print(s); assert(s == 501500); }

// @strict-null false
class C { x: number = 5; v: number[] = []; m() { return this.x; } }
class H { c: C | null = null; }
function mkH(): H { const r = new H(); return r; }
function mk(): C { const r = new C(); r.v.push(1); return r; }
function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) { const c = mk(); c.x = 999; c.v.push(99); junk.push(c); }
}
let G: H = mkH();
function f(x: C) { G.c = null; churn(); print(x.x); }
function g(h: H) { h.c = mk(); f(h.c); }
function main() { g(G); }

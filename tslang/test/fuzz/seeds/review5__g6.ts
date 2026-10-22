// @strict-null false
class C { x: number = 5; v: number[] = []; m() { return this.x; } }
class H { c: C | null = null; }
function mkH(): H { const r = new H(); return r; }
function mk(): C { const r = new C(); r.v.push(1); return r; }
function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) { const c = mk(); c.x = 999; c.v.push(99); junk.push(c); }
}
let G: C = mk();
function run(h: C) { G = mk(); churn(); print(h.x); }
function main() { G.x = 5; run(G); }

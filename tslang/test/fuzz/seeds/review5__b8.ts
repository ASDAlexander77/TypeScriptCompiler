// @strict-null false
class C { x: number = 5; v: number[] = []; m() { return this.x; } }
class H { c: C | null = null; }
function mkH(): H { const r = new H(); return r; }
function mk(): C { const r = new C(); r.v.push(1); return r; }
function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) { const c = mk(); c.x = 999; c.v.push(99); junk.push(c); }
}
function run(h: H) {
    const c = mk();
    h.c = c;
    try {
        h.c = null;
        return;
    } finally {
        churn();
        print(c.x);
    }
}
function main() { const h = mkH(); run(h); }

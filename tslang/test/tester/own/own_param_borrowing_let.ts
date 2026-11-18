// A local declared from a parameter and given nothing but borrows besides - `let m = p; if (c) m = k.b;`
// (TSLANG 56), or the parameter itself given an element (`p = q.items[0]`) - borrows the parameter too:
// the function does not keep it, and the caller still holds its argument after the call. Kept, nothing
// gave the argument back (160 MB at 2M calls under -mm=own, 42 MB now), and the caller could not read
// it after the call.
class V { x: number; items: V[] = []; constructor(x: number) { this.x = x; } }
class K { b: V = new V(5); f: V = new V(0); }
function churn() { const j: V[] = []; for (let i = 0; i < 64; i++) j.push(new V(999)); }
function useParam(q: V, p: V): number { if (p.x > 7) p = q.items[0]; return p.x; }
function reads(k: K, p: V, c: boolean): number { let m = p; if (c) m = k.b; return m.x; }
// fuzz 949146: a copy of the parameter on the other branch only reads it
function copyElse(q: V, p: V): number { if (p.x > 7) { p = q.items[0]; } else { const v = p; return v.x; } return p.x; }
function main() {
    const k = new K();
    for (let i = 0; i < 300; i++) {
        const p = new V(i);
        assert(reads(k, p, i % 2 == 0) == (i % 2 == 0 ? 5 : i), "reads");
        churn();
        assert(p.x == i, "argument still the caller's");
        const a = new V(i + 9); a.items.push(new V(1));
        assert(useParam(a, new V(i % 10)) == (i % 10 > 7 ? 1 : i % 10), "read");
        churn();
        assert(a.items[0].x == 1, "element still there");
        const c = new V(i % 10);
        assert(copyElse(a, c) == (i % 10 > 7 ? 1 : i % 10), "copy else");
        churn();
        assert(c.x == i % 10 && a.items[0].x == 1, "argument and element still there");
    }
    print("done.");
}

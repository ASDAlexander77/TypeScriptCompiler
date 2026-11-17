class T { v: number; constructor(v: number) { this.v = v; } }
class H { t: T | null = null; }
function f() {
    let q = new T(1); let r = new T(2); let b = q; const h = new H();
    h.t = q; h.t = null; print(b.v); b = r; print(b.v);
}
function main() { f(); print("done."); }

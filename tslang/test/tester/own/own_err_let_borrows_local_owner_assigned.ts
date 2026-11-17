class T { v: number; constructor(v: number) { this.v = v; } }
function f() {
    let q = new T(1); let r = new T(2); let b = q; b = r; r = new T(3); print(b.v);
}
function main() { f(); print("done."); }

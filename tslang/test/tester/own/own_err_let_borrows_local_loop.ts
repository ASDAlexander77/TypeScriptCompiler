class T { v: number; constructor(v: number) { this.v = v; } }
function f() {
    let q = new T(1); let r = new T(2); let b = q;
    for (let i = 0; i < 3; i++) { print(b.v); b = r; r = new T(i); }
}
function main() { f(); print("done."); }

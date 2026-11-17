class T { v: number; constructor(v: number) { this.v = v; } }
function f() {
    let q = new T(1); let r: T | null = new T(2); let qq = q && r; qq = r && q; r = null; print(qq.v);
}
function main() { f(); print("done."); }

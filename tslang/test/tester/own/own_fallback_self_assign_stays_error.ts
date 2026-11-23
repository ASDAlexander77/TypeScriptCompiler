// -mm=own --own-fallback=count: a field given its own value twice (`p.v = p.v; p.v = p.v`). The dry
// run erases the first read's retain with the second store's release of the old value (a balanced
// pair), and refuses only the second read. Counted, that read kept its retain while the release
// stayed erased, and the first store's release - which rc's erased retain held up - freed the object
// the field still held (fuzz 1123296). Its store's release is another value's: not counted, the
// error stays, at its place.
class V { x: number; constructor(x: number) { this.x = x; } }
class N { v: V; constructor(v: V) { this.v = v; } }
@noinline function mk(i: number): N { return new N(new V(i)); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(mk(777)); } }
function twice(p: N): number {
    p.v = p.v;
    p.v = p.v;
    churn();
    return p.v.x;
}
function main() {
    let s: number = 0;
    for (let i = 0; i < 20; i++) { s += twice(mk(i + 5)); }
    print(s);
    print("done.");
}

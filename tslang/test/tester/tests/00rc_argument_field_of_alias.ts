// A call given an object and a field of what may be the same object, read through another name:
// `f2(HG.n, n11.v)` where the local `n11` was given `HG.n`. `f2` sets `p3.v`, which under rc freed the
// block its other parameter, read out of that field, still pointed at (fuzz seed 937072). rc holds
// an argument read out of a field of an object of the same class as another argument's.
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; constructor(x: number) { this.x = x; this.w = new W(x); } }
class N { v: V; constructor(v: V) { this.v = v; } }
class H { n: N = new N(new V(0)); }
@noinline function mv(i: number): V { return new V(i); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(new V(777))); } }
const HG = new H();
function f2(p3: N, p4: V): number {
    p3.v = mv(9);
    churn();
    return p4.w.x;
}
function main() {
    let s: number = 0;
    for (let i = 0; i < 100; i++) {
        const n11 = HG.n;
        s += f2(HG.n, n11.v);
        HG.n.v = new V(0);
    }
    assert(s == 0, "the field's old value read after the callee replaced it");
    print("done.");
}

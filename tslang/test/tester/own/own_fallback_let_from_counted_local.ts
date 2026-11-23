// -mm=own --own-fallback=count: a `let` that borrows a global's field (counted: it cannot be stored),
// and a `const` declared from it. rc retains for the `const` and releases it; own erased the retain
// as a move out of the `let` and kept the release, and the field's object was freed while the
// global still held it (fuzz 1124384). The `const` is in the counted `let`'s group: its ops stay.
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; constructor(x: number) { this.x = x; this.w = new W(x); } }
class N { v: V; items: V[]; constructor(v: V) { this.items = [new V(v.x)]; this.v = v; } }
class H { n: N = new N(new V(0)); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(new V(777))); } }
const HG = new H();
function main() {
    let s: number = 0;
    for (let j = 0; j < 20; j++) {
        let n15: N = HG.n;
        const n16: N = n15;
        s += n16.v.x;
        churn();
    }
    churn();
    assert(HG.n.items[0].x == 0 && HG.n.v.w.x == 0, "the global's field is still alive");
    print(s, HG.n.items[0].x);
    print("done.");
}

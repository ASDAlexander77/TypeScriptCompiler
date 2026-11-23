// -mm=own --own-fallback=count: a `let` that borrows an element of a global's array and is then given
// a fresh value (TSLANG 66). Own repairs it - the fresh values go into a holder of their own, and the
// `let` borrows - and accepts it, so the flag must decide it the same way: nothing counted, no warning.
// Counted as well, its rc ops were no longer whole (the holder took the release of its assignment) and
// the element it borrowed was freed while the array still held it (fuzz 1122083).
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; constructor(x: number) { this.x = x; this.w = new W(x); } }
class N { items: V[]; constructor(x: number) { this.items = [new V(x)]; } }
@noinline function mv(i: number): V { return new V(i); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(777)); } }
let G: N = new N(4);
function main() {
    let s: number = 0;
    for (let i = 0; i < 300; i++) {
        let v: V = G.items[0];
        s += v.x;
        v = mv(i + 1);
        s += v.w.x;
        churn();
    }
    assert(G.items[0].x == 4 && G.items[0].w.x == 4, "the element is still the array's");
    print(s, G.items[0].x);
    print("done.");
}

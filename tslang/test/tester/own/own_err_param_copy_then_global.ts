// A parameter copied into a local and then given a global's field: the local takes the argument, so
// the function keeps it, and the caller may not give it a borrow of a global (fuzz 952414: treated as a
// borrow, the copy freed the caller's object).
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; a: number[]; constructor(x: number) { this.x = x; this.w = new W(x); this.a = [x]; } get(): number { return this.x; } }
class N { v: V; items: V[]; constructor(v: V) { this.items = [new V(v.x)]; this.v = v; } }
class H { n: N = new N(new V(0)); }
@noinline function mk(i: number): N { return new N(new V(i)); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(new V(777))); } }
const HG = new H();
function f1(p2: N, p3: N): number {
    let acc: number = 0;
    const n4 = p3;
    p3 = HG.n;
    return acc + p3.items[0].x;
}
function main() {
    let s: number = 0;
    s += f1(mk(1), (HG.n = mk(9), HG.n));
    let n18: N = HG.n;
    churn();
    print("c2", n18.v.x);
    assert(n18.v.x == 9, "c2");
}

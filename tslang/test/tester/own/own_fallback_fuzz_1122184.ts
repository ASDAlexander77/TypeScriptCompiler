// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): fuzz
// program 1122184, accepted only with the flag, its counted values (a borrow of a field or an element
// stored) neither leak nor are freed early. The fuzz program also used an array literal no local holds
// (`[a, b][1]`), which own leaks without the flag too; that line is replaced here (see own-readiness.md).
class W { x: number; constructor(x: number) { this.x = x; } }
class V { x: number; w: W; a: number[]; constructor(x: number) { this.x = x; this.w = new W(x); this.a = [x]; } get(): number { return this.x; } }
class N { v: V; items: V[]; constructor(v: V) { this.items = [new V(v.x)]; this.v = v; } }
class H { n: N = new N(new V(0)); }
interface IX { v: V; }
interface IY { v: V; items: V[]; }
interface IG { get(): number; }
@noinline function mk(i: number): N { return new N(new V(i)); }
@noinline function mv(i: number): V { return new V(i); }
function churn() { const hold: N[] = []; for (let i = 0; i < 64; i++) { hold.push(new N(new V(777))); } }
class T { s: string; constructor(s: string) { this.s = s; } }
@noinline function ms(i: number): string { return "m" + i; }
@noinline function mt(i: number): T { return new T("t" + i); }
const HG = new H();
const GA: N[] = [mk(6)];
@noinline
function g1(p2: N[], p3: N): N {
    let acc: number = 0;
    p2[0] = mk(8);
    return p3;
}
function main() {
    let s: number = 0;
    if (GA[GA.length - 1].v.x > 5) {
        GA[0] = HG.n;
    }
    GA[0] = mk(7);
    for (let i = 0; i < 300; i++) {
        let st4: string = "s" + i + 6;
        HG.n = HG.n;
        const sa5: string[] = ["s" + 8 + "k" + i + 3 + "k" + 4, `t${st4.length}`, ms(5)];
    }
    churn();
    print("c0", s);
    print("c1", HG.n.v.x);
    print("c2", GA[0].items[0].x);
    print("done.");
}

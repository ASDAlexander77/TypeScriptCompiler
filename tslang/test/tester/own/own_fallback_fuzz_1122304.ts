// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): fuzz
// program 1122304, accepted only with the flag, its counted values (a borrow of a field or an element
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
let GS: Shared<V> = new Shared<V>(mv(3));
@noinline
function f1(p2: N): number {
    let acc: number = 0;
    let sa3: string[] = [ms(2), "s" + 6];
    sa3.push(sa3[0]);
    GS.value = p2.items[0];
    return acc + GS.value.x;
}
function main() {
    let s: number = 0;
    s += f1(mk(2));
    s += f1(mk(9));
    for (let i = 0; i < 300; i++) {
        let sa4: string[] = [ms(i + 9), "s" + 3];
        sa4[0] = `t${GS.value.w.x}`;
        sa4[0] = sa4[0];
    }
    churn();
    print("c0", s);
    print("c1", GS.value.x);
    print("done.");
}

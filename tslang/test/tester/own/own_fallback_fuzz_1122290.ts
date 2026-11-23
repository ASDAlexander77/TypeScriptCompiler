// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): fuzz
// program 1122290, accepted only with the flag, its counted values (a borrow of a field or an element
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
@noinline
function f1(p2: H, p3: N): number {
    let acc: number = 0;
    p2.n = p2.n;
    return acc + p3.items[0].x;
}
function main() {
    let s: number = 0;
    s += 9;
    s += f1(new H(), mk(9));
    for (let i = 0; i < 100; i++) {
        s += 5;
        for (let j4 = 0; j4 < 3; j4++) {
            let st5: string = "s" + i + 4;
        }
        s += f1(new H(), mk(i + 2));
        try {
            for (let j6 = 0; j6 < 2; j6++) {
                let v7: V = mv(i + 8);
            }
            if (i % 2 == 0) {
                s += 1;
                churn();
                s += 7;
            }
            s += f1(new H(), mk(6));
        } finally {
            s += f1(new H(), mk(2));
        }
        if (9 > 6) { continue; }
    }
    churn();
    print("c0", s);
    print("done.");
}

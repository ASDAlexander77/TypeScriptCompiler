// -mm=own --own-fallback=count, the leak gate (mm_fuzz.py leaks --own-fallback over own_fallback_*.ts): fuzz
// program 1122123, accepted only with the flag. A counted value is put into an array literal given straight
// to a call (`f2([p6.n], ...)`): the literal is given back after the call (it leaked before), and with it
// the count the literal held, while the value's other owner keeps it. Its indexed literals (`[a, b][1]`)
// are given back too.
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
function r1(acc: N, x: N): N { return mk(acc.v.x + 1); }
@noinline
function f2(p3: N[], p4: N): number {
    let acc: number = 0;
    p4 = p3[0];
    return acc + p4.v.a[0];
}
@noinline
function f5(p6: H, p7: N): number {
    let acc: number = 0;
    acc += p6.n.items[0].x;
    p7 = p6.n;
    p7 = mk(1);
    let fs8: (n: number) => string = (k: number): string => "c" + k + "k" + 9;
    acc += f2([p6.n], (p7 = mk(3), p6.n));
    churn();
    return acc + p7.v.get();
}
function g9(p10: N): N {
    let acc: number = 0;
    acc += f2([mk(5), mk(4)], (p10 = mk(9), p10));
    acc += p10.v.get();
    let a11: N[] = [mk(6)];
    churn();
    return a11[0];
}
function main() {
    let s: number = 0;
    try {
        for (let j12 = 0; j12 < 2; j12++) {
            for (let j13 = 0; j13 < 4; j13++) {
                s += f5(new H(), mk(7));
                s += 3;
                s += 2;
                let n14: N = g9(mk(5));
            }
            s += 3;
        }
    } finally {
        s += 1;
        const fs15: (n: number) => string = (k: number): string => "c" + k + "k" + 6;
    }
    const n16: N = g9(mk(1));
    let st17: string = [[`t${n16.v.a[0]}`, "k" + 9 + "k" + 4][0], [`t${n16.v.a[0]}`, ["k" + 4, "k" + 6][1]][0]][1];
    for (let i = 0; i < 1; i++) {
        s += st17.length;
        const st18 = ms(1);
    }
    churn();
    print("c0", s);
    print("c1", n16.v.x);
    print("c2", st17.length);
    print("done.");
}

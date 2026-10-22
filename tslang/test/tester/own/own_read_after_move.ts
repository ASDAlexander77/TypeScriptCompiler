// @strict-null false
// -mm=own: a value moved into a new owner - a field, an owning local - may still be read after the
// move while that owner holds it unchanged: the read borrows it through the owner (spec 29). A path
// from the move to the read that overwrites, releases or moves the owner on stays an error.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
class N { v: number[] = []; }
class C { x: number = 5; }
class H { c: C | null = null; }
// another class's field of the same type at the same position is matched as the same field
// (dropsChain), so the loop's holder keeps its field elsewhere
class HW { k: number = 0; c: C | null = null; }
function churn() {
    let s: number = 0;
    for (let i = 0; i < 64; i++) {
        const junk = new V(999);
        const c = new C();
        c.x = 999;
        const n = new N();
        n.v.push(99);
        s += junk.x + c.x + n.v.length;
    }

    return s;
}
function main() {
    const a: number[] = [1];
    const n = new N();
    n.v = a;
    print(a.length);                       // 1
    const h = new H();
    const c = new C();
    h.c = c;
    print(c.x);                            // 5
    const vs = [new V(1), new V(3), new V(2)];
    let best: V = undefined;
    let s: number = 0;
    for (let i in vs) {
        let v = V.plus(vs[i], vs[i]);
        best = v;
        s += v.x;                          // read after the move into best
    }
    print(best.x, s);                      // 4 12
    let t: number = 0;
    for (let k = 0; k < 100000; k++) {
        const w = new V(k);
        const hw = new HW();
        const cw = new C();
        hw.c = cw;
        best = w;
        t += w.x - k + cw.x;
    }

    // a declaration takes the value over; `p` is read through `q`, then given a new value of its own
    let p = new V(7);
    let q = p;
    print(p.x);                            // 7
    p = new V(8);
    churn();
    print(q.x, p.x);                       // 7 8
    assert(q.x == 7 && p.x == 8);
    print(a.length, a[0], c.x, best.x, s, t);   // 1 1 5 99999 12 500000
    assert(a.length == 1 && a[0] == 1 && c.x == 5 && best.x == 99999 && s == 12 && t == 500000);
    assert(n.v.length == 1 && h.c.x == 5);
    print("done.");
}

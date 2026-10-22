// @strict-null false
// -mm=own: a value moved into a new owner - a field, an element, an owning local - may still be read
// after the move while that owner holds it unchanged: the read borrows it through the owner (spec
// 29). A path from the move to the read that overwrites, releases or moves the owner on stays an
// error. The values come from factories, so that stack promotion leaves them heap blocks.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
class N { v: number[] = []; }
class C { x: number = 5; }
class H { c: C | null = null; }
// another class's field of the same type at the same position is matched as the same field
// (dropsChain), so the loop's holder and K keep their fields elsewhere
class HW { k: number = 0; c: C | null = null; }
class K {
    k: number = 0;
    k2: number = 0;
    f: C | null = null;
    put(n: number) {
        const c = mkc(n);
        this.f = c;                        // moved into a field of `this`
        churn();
        return c.x;
    }
}
function mka(n: number) {
    const r: number[] = [];
    r.push(n);
    return r;
}
function mkc(n: number) {
    const r = new C();
    r.x = n;
    return r;
}
function mkv(n: number) {
    return new V(n);
}
function churn() {
    let s: number = 0;
    for (let i = 0; i < 64; i++) {
        const junk = mkv(999);
        const c = mkc(999);
        const n = new N();
        n.v.push(99);
        s += junk.x + c.x + n.v.length;
    }

    return s;
}
function main() {
    const a = mka(1);
    const n = new N();
    n.v = a;
    print(a.length);                       // 1
    const h = new H();
    const c = mkc(5);
    h.c = c;
    print(c.x);                            // 5
    const arr: C[] = [mkc(0)];
    const ce = mkc(8);
    arr[0] = ce;                           // moved into an element
    churn();
    print(ce.x);                           // 8
    const k = new K();
    const kx = k.put(6);
    print(kx, k.f.x);                      // 6 6
    const vs = [mkv(1), mkv(3), mkv(2)];
    let best: V = undefined;
    let s: number = 0;
    for (let i in vs) {
        let v = V.plus(vs[i], vs[i]);
        best = v;
        s += v.x;                          // read after the move into best
    }
    print(best.x, s);                      // 4 12
    let t: number = 0;
    for (let j = 0; j < 100000; j++) {
        const w = mkv(j);
        const hw = new HW();
        const cw = mkc(5);
        hw.c = cw;
        best = w;
        t += w.x - j + cw.x;
    }

    // a declaration takes the value over; `p` is read through `q`, then given a new value of its own
    let p = mkv(7);
    let q = p;
    print(p.x);                            // 7
    p = mkv(8);
    churn();
    print(q.x, p.x);                       // 7 8
    assert(q.x == 7 && p.x == 8);
    print(a.length, a[0], c.x, ce.x, best.x, s, t);   // 1 1 5 8 99999 12 500000
    assert(a.length == 1 && a[0] == 1 && c.x == 5 && ce.x == 8);
    assert(best.x == 99999 && s == 12 && t == 500000);
    assert(n.v.length == 1 && h.c.x == 5 && arr[0].x == 8 && kx == 6 && k.f.x == 6);
    print("done.");
}

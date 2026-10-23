// -mm=own: a field read through an interface is a place read like a class field's (spec 28). A
// `let` initialized from one - through any chain of field reads and interface views - borrows the
// field, bounded by what it was read through; a call ends the borrow only when its facts say it
// may drop or overwrite it. A function may return such a read as a borrow of its parameter.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
interface R { dir: V; }
interface Is { ray: R; dist: number; }
class RayC implements R { constructor(public dir: V) {} }
function f(r: R) { let d = r.dir; let p = V.plus(d, d); return V.plus(p, d); }
function sh(i: Is) { let d = i.ray.dir; let pos = V.plus(d, i.ray.dir); return V.plus(d, pos); }
function dirOf(r: R) { return r.dir; }
function keep(r: R, v: V) { r.dir = v; }
function main() {
    const r: R = { dir: new V(1) };
    const i: Is = { ray: { dir: new V(2) }, dist: 3 };
    print(f(r).x, sh(i).x);   // 3 6
    const c = new RayC(new V(4));
    print(f(c).x, dirOf(c).x);   // 12 4
    let s: number = 0;
    for (let k = 0; k < 100000; k++) {
        s += f(r).x + sh(i).x + dirOf(i.ray).x;
    }

    print(s, r.dir.x, i.ray.dir.x, c.dir.x);   // 1.1e+06 1 2 4
    assert(s == 1100000 && r.dir.x == 1 && i.ray.dir.x == 2 && c.dir.x == 4);
    // a store through an interface field keeps the value: `nv` moves into `k.dir`
    const k: R = { dir: new V(0) };
    const nv = new V(7);
    keep(k, nv);
    let t: number = 0;
    for (let j = 0; j < 1000; j++) {
        t += V.plus(k.dir, k.dir).x;
    }

    print(t, k.dir.x);   // 14000 7
    assert(t == 14000 && k.dir.x == 7);
    print("done.");
}

// -mm=own: an object literal that owns its fields - fresh values, or locals moved in - is a value
// like a fresh object. It moves into a call argument (the caller releases it after a callee that
// borrows it; one that keeps it takes it), an array literal's element, a nested literal's field, a
// local through an interface view, or the result (spec 27).
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
interface Ray { start: V; dir: V; }
interface L { pos: V; }
interface Sc { ls: V[]; c: V; }
class Keeper { rays: Ray[] = []; }
function len(r: Ray) { return r.start.x + r.dir.x; }
function keepRay(k: Keeper, r: Ray) { k.rays.push(r); }
function mk(): Sc { return { ls: [new V(1), new V(2)], c: new V(3) }; }
function main() {
    const p = new V(1);
    const d = new V(2);
    print(len({ start: p, dir: d }));                          // 3   (p, d move in)
    const q = new V(1);
    print(len({ start: V.plus(q, q), dir: q }));               // 3   (fresh + moved)
    const ls: L[] = [{ pos: new V(1) }, { pos: new V(2) }];
    const r: Ray = { start: new V(4), dir: new V(5) };
    const s = mk();
    print(ls[1].pos.x, r.dir.x, s.c.x, s.ls[1].x);             // 2 5 3 2
    const k = new Keeper();
    for (let i = 0; i < 1000; i++) keepRay(k, { start: new V(i), dir: new V(1) });
    print(k.rays.length, k.rays[999].start.x);                 // 1000 999
    print("done.");
}

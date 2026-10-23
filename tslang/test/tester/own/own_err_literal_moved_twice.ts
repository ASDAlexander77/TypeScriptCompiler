// -mm=own rejects: `p` moves into the first record, which `keepRay` keeps, and is used again in
// the second (spec 27).
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
}
interface Ray { start: V; dir: V; }
class Keeper { rays: Ray[] = []; }
function keepRay(k: Keeper, r: Ray) { k.rays.push(r); }
function main() {
    const p = new V(1);
    const k = new Keeper();
    keepRay(k, { start: p, dir: new V(2) });
    keepRay(k, { start: p, dir: new V(3) });
    print(k.rays.length);
}

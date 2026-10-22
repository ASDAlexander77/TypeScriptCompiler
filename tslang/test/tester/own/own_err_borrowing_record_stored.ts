// @strict-null false
// -mm=own rejects: `sc.ts[0].hit(ray)` is a borrowing record (spec 32) that holds `sc.ts[0]` and
// `ray`; stored into a global's field it would outlive both.
class V {
    constructor(public x: number) {}
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 200; i++) {
        keep.push(new V(77777));
    }
    return keep.length;
}

interface Ray { start: V; dir: V; }
interface Hit { thing: Th; ray: Ray; dist: number; }
interface Th { hit(ray: Ray): Hit; k: number; }
class A implements Th { k: number = 1; hit(ray: Ray): Hit { return { thing: this, ray: ray, dist: ray.dir.x }; } }
class Sc { ts: Th[] = [new A()]; }
class Keep { h: Hit = null; }

const keep = new Keep();

function store(ray: Ray, sc: Sc) {
    keep.h = sc.ts[0].hit(ray);
}

function main() {
    const sc = new Sc();
    store({ start: new V(1), dir: new V(2) }, sc);
    churn();
    print(keep.h.ray.start.x);
}

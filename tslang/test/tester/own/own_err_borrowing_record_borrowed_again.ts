// -mm=own rejects: `id(h)` is a borrow of the record `h` (spec 32), so it may not be read after what
// `h` borrows - `a` - is given a new value, its old one freed.
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

function id(r: Hit): Hit { return r; }

function main() {
    let a = new A();
    const ray: Ray = { start: new V(1), dir: new V(2) };
    let h = a.hit(ray);
    const h2 = id(h);
    a = new A();
    churn();
    print(h2.thing.k);
}

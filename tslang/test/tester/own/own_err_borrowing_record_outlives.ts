// -mm=own rejects: `h` is a borrowing record (spec 32) that holds `a`, and `a` is given a new value
// - its old one freed - before `h` is read.
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

function main() {
    let a = new A();
    const ray: Ray = { start: new V(1), dir: new V(2) };
    let h = a.hit(ray);
    a = new A();
    churn();
    print(h.thing.k);
}

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
interface HasDist { dist: number; }
function main() {
    const a = new A();
    const ray: Ray = { start: new V(1), dir: new V(2) };
    const h = a.hit(ray);
    const d: HasDist = h;
    print(d.dist);
}

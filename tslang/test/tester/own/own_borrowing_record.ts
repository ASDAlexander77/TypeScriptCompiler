// @strict-null false
// -mm=own: a method that returns an object literal holding `this` and its parameter returns a
// borrowing record (spec 32): the caller owns the box, which may not outlive the receiver and the
// argument, and whose release frees only what the record owns - here nothing but itself. A call
// through a closed interface gets the record's facts; `closest` keeps the best record in a local
// (moved there on some paths only) and returns it, so its own result borrows its `ray` and `sc`.
// churn() keeps what it allocates alive, so a block freed too early is reused before it is read.
class V {
    constructor(public x: number) {}
    static plus(a: V, b: V) { return new V(a.x + b.x); }
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
class B implements Th {
    k: number = 2;
    hit(ray: Ray): Hit {
        if (ray.dir.x > 5) return null;
        return { thing: this, ray: ray, dist: ray.dir.x * 2 };
    }
}
class Sc { ts: Th[] = [new A(), new B()]; }

function closest(ray: Ray, sc: Sc) {
    let best = +Infinity;
    let bestHit: Hit = undefined;
    for (let i in sc.ts) {
        let h = sc.ts[i].hit(ray);
        if (h != null && h.dist < best) { bestHit = h; best = h.dist; }
    }
    return bestHit;
}

function main() {
    const sc = new Sc();
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        const h = closest({ start: new V(0), dir: new V(i % 10) }, sc);
        churn();
        s += h.thing.k + h.dist + h.ray.start.x;
        assert(h.ray.dir.x == i % 10);
    }

    print(s);        // 5500: A always wins (dist d against B's 2d), k 1
    assert(s == 5500);
    print("done.");
}

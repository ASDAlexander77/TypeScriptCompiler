// -mm=own: the ray tracer's `intersections` loop over @value records. `closest` starts undefined
// and is given the nearer of each `t.hit(ray)` inside the loop; after the loop it is returned, and
// the caller passes it to `shade`, which reads `thing` and calls through it. Each record borrows the
// receiver it was made by, an element of the array the caller owns. churn() between the loop and
// each read reuses the blocks a freed `S` would leave.
@value
class Vector {
    constructor(public readonly x: number, public readonly y: number) {}
}

@value
interface Ray {
    readonly start: Vector;
    readonly dir: Vector;
}

interface Thing {
    hit(r: Ray): Hit | null;
    id(): int;
}

@value
interface Hit {
    readonly thing: Thing;
    readonly ray: Ray;
    readonly dist: number;
}

class S implements Thing {
    constructor(private n: int, private at: number) {}

    hit(r: Ray): Hit | null {
        const d = this.at - r.start.x;
        return d > 0 ? { thing: this, ray: r, dist: d } : null;
    }

    id(): int {
        return this.n;
    }
}

class Scene {
    things: Thing[] = [new S(1, 5), new S(2, 3), new S(3, 8)];
}

function churn() {
    const keep: S[] = [];
    for (let i = 0; i < 64; i++) {
        keep.push(new S(77777, 77777));
    }

    return keep.length;
}

class Tracer {
    intersections(ray: Ray, scene: Scene) {
        let best = +Infinity;
        let closest: Hit | undefined = undefined;
        for (let i = 0; i < scene.things.length; i++) {
            const inter = scene.things[i].hit(ray);
            if (inter != null && inter.dist < best) {
                closest = inter;
                best = inter.dist;
            }
        }

        return closest;
    }

    shade(h: Hit): number {
        return h.thing.id() * 100 + h.dist;
    }

    trace(ray: Ray, scene: Scene): number {
        const h = this.intersections(ray, scene);
        churn();
        if (h === undefined) {
            return -1;
        }

        return this.shade(h);
    }
}

function main() {
    const scene = new Scene();
    const tracer = new Tracer();
    let total: number = 0;
    for (let i = 0; i < 1000; i++) {
        total += tracer.trace({ start: new Vector(i % 10, 0), dir: new Vector(1, 0) }, scene);
    }

    print(total);
    // start x 0..2: S2 (at 3) is nearest, dist 3 - x; 3..4: S1 (at 5), dist 5 - x; 5..7: S3 (at 8),
    // dist 8 - x; 8..9: none (-1). Per ten: 3*200 + 6 + 2*100 + 3 + 3*300 + 6 - 2 = 1713
    assert(total == 171300);
    print("done.");
}

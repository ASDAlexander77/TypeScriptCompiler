// -mm=own: a @value record holding a borrow of `this` (the ray tracer's Intersection), returned as
// `Hit | null` by methods reached through the interface `Thing`, from a ternary (`S`) and from an
// `if` (`P`). The record is a value, a tuple: there is no box, and its `thing` borrows the
// receiver, which the caller's array owns. The caller reads `h.thing.id()` through the interface and
// `h.dist` after churn(), which reuses blocks of the size a freed `S` or `P` would leave.
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
    constructor(private n: int) {}

    hit(r: Ray): Hit | null {
        return r.dir.x > 0 ? { thing: this, ray: r, dist: r.dir.x } : null;
    }

    id(): int {
        return this.n;
    }
}

class P implements Thing {
    constructor(private n: int) {}

    hit(r: Ray): Hit | null {
        if (r.dir.y > 0) {
            return { thing: this, ray: r, dist: r.dir.y * 2 };
        }

        return null;
    }

    id(): int {
        return this.n;
    }
}

function churn() {
    const keep: S[] = [];
    for (let i = 0; i < 64; i++) {
        keep.push(new S(77777));
    }

    return keep.length;
}

function main() {
    const things: Thing[] = [new S(1), new P(2)];
    let total: number = 0;
    let missing = 0;
    for (let i = 0; i < 1000; i++) {
        const ray: Ray = { start: new Vector(0, 0), dir: new Vector((i % 3) - 1, (i % 5) - 2) };
        for (let k = 0; k < things.length; k++) {
            const h = things[k].hit(ray);
            churn();
            if (h != null) {
                total += h.thing.id() + h.dist + h.ray.start.x;
            } else {
                missing++;
            }
        }
    }

    print(total, missing);
    // S hits 333 times (id 1, dist 1); P 200 times with dist 2 and 200 with dist 4 (id 2)
    assert(total == 2666);
    assert(missing == 1267);
    print("done.");
}

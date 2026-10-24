// The two checks of value_interface.ts that -mm=own rejects (own spec 36, "Not here"); every other
// model runs them. A literal nested in another, holding new vectors and a new T1, out of order: an
// owning tuple moved into a tuple's field (#679's limit). A record that borrows two temporaries,
// converted to an ordinary interface: the box made of it would own what the record borrows.
class Vec {
    constructor(public x: number, public y: number) {}
}

interface Thing {
    id(): int;
}

class T1 implements Thing {
    id(): int {
        return 7;
    }
}

@value
interface Ray {
    readonly start: Vec;
    readonly dir: Vec;
}

@value
interface Hit {
    readonly thing: Thing;
    readonly ray: Ray;
    readonly dist: number;
}

interface HasStart {
    start: Vec;
}

function mk(s: Vec, d: Vec): Ray {
    return { start: s, dir: d };
}

function main() {
    // nested two levels, each out of order
    const n: Hit = { ray: { dir: new Vec(0, 3), start: new Vec(5, 0) }, dist: 1, thing: new T1() };
    assert(n.ray.start.x == 5 && n.ray.dir.y == 3 && n.thing.id() == 7);

    // converted to an ordinary interface
    const s: HasStart = mk(new Vec(6, 0), new Vec(0, 0));
    assert(s.start.x == 6);

    print("done.");
}

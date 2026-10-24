// `@value interface`: a literal of the interface's type is a value - a tuple of its fields in
// declaration order, with no box and no vtable. It is read, passed, nested and converted.
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

// the fields are written in another order than the interface's
function hit(t: Thing, r: Ray, d: number): Hit {
    return { dist: d, ray: r, thing: t };
}

function len(r: Ray) {
    return r.start.x + r.dir.y;
}

function main() {
    assert(len({ start: new Vec(1, 2), dir: new Vec(3, 4) }) == 5);

    const h = hit(new T1(), mk(new Vec(1, 0), new Vec(0, 2)), 9);
    assert(h.dist == 9 && h.thing.id() == 7 && h.ray.dir.y == 2);

    // nested two levels, each out of order
    const n: Hit = { ray: { dir: new Vec(0, 3), start: new Vec(5, 0) }, dist: 1, thing: new T1() };
    assert(n.ray.start.x == 5 && n.ray.dir.y == 3);

    // converted: to `any`, to an ordinary interface, to a type literal and back
    // tslang reads no member of an `any`: the value is cast back to read it
    const a: any = mk(new Vec(8, 0), new Vec(0, 0));
    assert((<Ray>a).start.x == 8);

    const s: HasStart = mk(new Vec(6, 0), new Vec(0, 0));
    assert(s.start.x == 6);

    const tl: { start: Vec; dir: Vec } = mk(new Vec(2, 0), new Vec(0, 4));
    const back: Ray = tl;
    assert(back.start.x == tl.start.x && back.dir.y == tl.dir.y);
    assert(back.start.x == 2 && back.dir.y == 4);

    assert(typeof mk(new Vec(0, 0), new Vec(0, 0)) == "object");

    let total: number = 0;
    for (let i = 0; i < 10000; i++) {
        total += len(mk(new Vec(i, 0), new Vec(0, 1)));
    }

    assert(total == 50005000);

    print("done.");
}

// Under rc, a field of a @value value held in an ordinary class's field, read into a const or into a
// literal of the @value interface, is retained the way the same read of a type literal is
// (value-rc-field-retain.cmake compares each pair of functions' retains and releases).
class Vec {
    constructor(public x: number) {}
}

@value
interface Ray {
    readonly start: Vec;
    readonly dir: Vec;
}

type RayT = { start: Vec; dir: Vec };

class Hit {
    r: Ray;

    constructor(r: Ray) {
        this.r = r;
    }
}

class HitT {
    r: RayT;

    constructor(r: RayT) {
        this.r = r;
    }
}

function rayFromHit(h: Hit) {
    const o: Ray = { start: h.r.start, dir: h.r.dir };
    return o.start.x + o.dir.x;
}

function rayTFromHit(h: HitT) {
    const o: RayT = { start: h.r.start, dir: h.r.dir };
    return o.start.x + o.dir.x;
}

function startFromHit(h: Hit) {
    const s = h.r.start;
    h.r = { start: new Vec(5), dir: new Vec(6) };
    return s.x;
}

function startTFromHit(h: HitT) {
    const s = h.r.start;
    h.r = { start: new Vec(5), dir: new Vec(6) };
    return s.x;
}

function main() {
    print(rayFromHit(new Hit({ start: new Vec(1), dir: new Vec(2) })));
    print(rayTFromHit(new HitT({ start: new Vec(1), dir: new Vec(2) })));
    print(startFromHit(new Hit({ start: new Vec(1), dir: new Vec(2) })));
    print(startTFromHit(new HitT({ start: new Vec(1), dir: new Vec(2) })));
}

// A @value type may reach itself through an ordinary class: the class holds a reference, not the
// value in place. Declared before and after the class.
class Box {
    constructor(public l: L | null) {}
}

@value
interface L {
    readonly v: number;
    readonly box: Box;
}

@value
interface M {
    readonly v: number;
    readonly box: Box2;
}

class Box2 {
    constructor(public m: M | null) {}
}

function main() {
    const a: L = { v: 1, box: new Box(null) };
    const b: L = { v: 2, box: new Box(a) };
    const l = b.box.l;
    assert(l !== null && l.v == 1);
    const c: M = { v: 3, box: new Box2(null) };
    const d: M = { v: 4, box: new Box2(c) };
    const m = d.box.m;
    assert(m !== null && m.v == 3);
    print("done.");
}

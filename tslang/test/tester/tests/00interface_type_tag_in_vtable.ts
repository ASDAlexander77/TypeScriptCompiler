// An interface is { vtable, this }: the type tag the reference-counting models retain and release
// the object by is the vtable's first slot, not a third field of the interface. Every kind of
// vtable carries it - a class's (and a subclass's own, whose object holds more), an object
// literal's, one laid out for an interface the value's type does not extend (#666), and the null
// vtable, whose tag is null and releases nothing. Each object is reached only through interfaces
// once its class reference is gone; a count taken or dropped through a wrong tag frees it early,
// and the reads after other blocks are made see what reuses its memory.
// Fields are read before any method call: one through a wrong vtable runs whatever it lands on.
interface Shape {
    id: number;
    area(): number;
}

interface HasId {
    id: number;
}

class Box {
    constructor(public w: number, public h: number) {}
}

class Rect implements Shape {
    constructor(public id: number, public box: Box) {}

    area(): number {
        return this.box.w * this.box.h;
    }
}

class Square extends Rect {
    constructor(id: number, side: number, public label: Box) {
        super(id, new Box(side, side));
    }
}

@noinline function mkRect(id: number, w: number, h: number): Shape {
    return new Rect(id, new Box(w, h));
}

@noinline function mkSquare(id: number, side: number): Shape {
    return new Square(id, side, new Box(id, id));
}

@noinline function mkLiteral(id: number, w: number): Shape {
    const box = new Box(w, 1);
    return { id: id, area: () => box.w };
}

@noinline function mkNull(): Shape | null {
    return null;
}

// blocks the size of what was freed, written over it if it was
@noinline function churn(): number {
    let sum: number = 0;
    for (let i = 0; i < 64; i++) {
        const b = new Box(-1, -1);
        const r = new Rect(-1, b);
        sum += r.id;
    }

    return sum;
}

function keep(shapes: Shape[]): HasId[] {
    const ids: HasId[] = [];
    for (const s of shapes) {
        ids.push(s);
    }

    return ids;
}

function main() {
    let shapes: Shape[] = [mkRect(1, 2, 3), mkSquare(2, 4), mkLiteral(3, 5)];
    assert(shapes[0].id == 1, "a class object's field");
    assert(shapes[1].id == 2, "a subclass object's field");
    assert(shapes[2].id == 3, "an object literal's field");

    // each seen through an interface its type does not extend, then the first views dropped
    const ids = keep(shapes);
    shapes = [];
    churn();

    assert(ids[0].id == 1, "a class object held through another interface");
    assert(ids[1].id == 2, "a subclass object held through another interface");
    assert(ids[2].id == 3, "an object literal held through another interface");

    // a copy, then the original given null: the copy holds the object
    let a: Shape | null = mkSquare(7, 3);
    const b: Shape | null = a;
    a = mkNull();
    churn();
    assert(b.id == 7, "a copy of an interface holds its object");
    assert(b.area() == 9, "and calls through its vtable");
    assert(a === null, "a null interface");

    // the null interface dropped in turn: its vtable's tag is null and releases nothing
    a = mkRect(8, 1, 1);
    churn();
    assert(a.id == 8, "given an object after null");
    assert(b.id == 7, "the copy still holds its object");

    print("done.");
}

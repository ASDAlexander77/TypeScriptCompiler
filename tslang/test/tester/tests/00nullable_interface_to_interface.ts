// An interface-or-null given to another interface-or-null whose interface it does not extend is the
// interface-to-interface cast (#666), which builds the target's vtable out of the value's own one. A
// plain copy kept the source's vtable, and a field read through the target read a method slot as the
// field's offset. A null value is null still, and its methods throw. Not under -mm=own, which does not
// take this program's interface values returned from calls yet.
interface Shape {
    id: number;
    area(): number;
}

interface HasId {
    id: number;
}

interface Sized {
    area(): number;
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

@noinline function mkSquare(id: number, side: number): Shape {
    return new Square(id, side, new Box(id, id));
}

@noinline function mkShape(flag: boolean): Shape | null {
    return flag ? mkSquare(5, 2) : null;
}

@noinline function mkNull(): HasId | null {
    return null;
}

// what a method of the interface does: -2 when it throws
function call(s: Sized | null): number {
    try {
        return s.area();
    } catch (e) {
        return -2;
    }
}

function main() {
    // a field first, and no method: a method called through a wrong vtable runs whatever code it lands on
    const b: Shape | null = mkSquare(7, 3);
    let n: HasId | null = mkNull();
    n = b;
    assert(n.id == 7, "a field read through the other interface");

    // members in another order: the field and the method each moved
    const s: Sized | null = b;
    assert(s.id == 7, "a field that moved");
    assert(call(s) == 9, "a method that moved");

    // a declaration, and a value that came back from a call
    const m: HasId | null = mkShape(true);
    assert(m.id == 5, "a returned value seen through the other interface");

    // null stays null, and its methods throw
    const nullShape: Shape | null = mkShape(false);
    const nullId: HasId | null = nullShape;
    assert(nullId === null, "null seen through the other interface");
    const nullSized: Sized | null = nullShape;
    assert(nullSized === null, "null seen through an interface with members in another order");
    assert(call(nullSized) == -2, "its method throws");

    print("done.");
}

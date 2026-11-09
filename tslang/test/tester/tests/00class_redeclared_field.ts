// A subclass that declares a field its base has (`id: number = 3` over Base's `id: number = 1`) has
// one property, as in TypeScript: its initializer sets Base's field, and a read through Base's type,
// or through an interface, sees 3. It had a slot of its own, so Base's still read 1.

interface HasId {
    id: number;
}

class Base {
    id: number = 1;
    name: string = "base";

    idOf() {
        return this.id;
    }
}

class Leaf extends Base {
    id: number = 3;
}

class Deeper extends Leaf {
    name: string = "deeper";
}

class Param extends Base {
    constructor(public id: number) {
        super();
    }
}

class Counter extends Base {
    id = 5;

    bump() {
        this.id++;
    }
}

function main() {
    const l = new Leaf();
    const b: Base = l;
    assert(l.id == 3, "the subclass's own type");
    assert(b.id == 3, "through the base's type");
    assert(b.idOf() == 3, "in a method of the base");

    const asId: HasId = b;
    assert(asId.id == 3, "through an interface");

    b.id = 4;
    assert(l.id == 4, "a write through the base, read through the subclass");

    const d: Base = new Deeper();
    assert(d.id == 3 && d.name == "deeper", "two levels down");

    const p: Base = new Param(7);
    assert(p.id == 7, "a parameter property");

    const c = new Counter();
    c.bump();
    const cb: Base = c;
    assert(cb.id == 6 && c.id == 6, "an untyped initializer, then an update");

    print("done.");
}

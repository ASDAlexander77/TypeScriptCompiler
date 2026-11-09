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

// declared again with a narrower type: still Base's slot, read through the subclass as its type

class Animal {
    name: string = "animal";
}

class Dog extends Animal {
    bark() {
        return "woof";
    }
}

class Holder {
    v: number | string = "s";
    owner: Animal = new Animal();
}

class NumberHolder extends Holder {
    v: number = 3;
    owner: Dog = new Dog();

    twice() {
        return this.v * 2;
    }
}

class Below extends NumberHolder {
}

interface HasNumber {
    v: number;
}

interface HasDog {
    owner: Dog;
}

function narrowed() {
    const n = new NumberHolder();
    const h: Holder = n;
    assert(n.v == 3, "the subclass's type");
    assert(h.v == 3, "through the base's type, the union");
    assert(n.owner.bark() == "woof", "a subclass's class type");
    assert(h.owner.name == "animal", "through the base's type, the base class");
    assert(n.twice() == 6, "arithmetic in a method of the subclass");

    n.v = 4;
    assert(h.v == 4, "a write through the subclass");
    n.v += 1;
    assert(n.v == 5 && h.v == 5, "a compound write");

    h.v = 7;
    assert(n.v == 7, "a write through the base");

    const asNumber: HasNumber = n;
    assert(asNumber.v == 7, "an interface wanting the narrow type");
    asNumber.v = 9;
    assert(n.v == 9 && h.v == 9, "a write through that interface");

    const withDog: HasDog = n;
    assert(withDog.owner.bark() == "woof", "an interface wanting the subclass's class type");
    withDog.owner = new Dog();
    assert(n.owner.bark() == "woof", "a write of an object through that interface");

    const below = new Below();
    assert(below.v + 1 == 4, "a class below the one that narrowed it");
}

function main() {
    narrowed();

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

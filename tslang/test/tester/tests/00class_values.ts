// A class as a value (`typeof A`): given, stored, put in an array, returned, held in a field, compared,
// and `new`-ed through - whichever class it holds is the one constructed, with its own constructor,
// field initializers and methods.
class Shape {
    name = "shape";

    constructor(public s: number) {}

    area() {
        return 0;
    }
}

class Square extends Shape {
    constructor(s: number) {
        super(s);
        this.name = "square";
    }

    area() {
        return this.s * this.s;
    }
}

// no constructor of its own: Shape's
class Triple extends Shape {
    area() {
        return 3 * this.s * this.s;
    }
}

class Plain {
    v = 5;
}

abstract class Base {}
class Impl extends Base {}

class Holder {
    constructor(public kind: typeof Shape) {}
}

class Counter {
    static count = 7;
}

// statics through a value: a subclass's static of the same name is the one called
class Unit {
    static label() {
        return "unit";
    }

    static scaled(k: number) {
        return k;
    }
}

class Double extends Unit {
    static label() {
        return "double";
    }

    static scaled(k: number) {
        return 2 * k;
    }
}

class Inherits extends Unit {}

interface Named {
    title(): string;
}

// an abstract class's own value: its vtable has its statics, its abstract and interface slots null
abstract class Kind implements Named {
    static n = 4;

    static id() {
        return 1;
    }

    abstract title(): string;
}

class Concrete extends Kind {
    static id() {
        return 2;
    }

    title() {
        return "concrete";
    }
}

function make(c: typeof Shape, s: number) {
    return new c(s);
}

function pick(i: number): typeof Shape {
    if (i == 0) return Square;
    if (i == 1) return Triple;
    return Shape;
}

function countOf(k: typeof Counter) {
    return k.count;
}

function labelOf(u: typeof Unit) {
    return u.label();
}

function main() {
    assert(make(Square, 3).area() == 9, "factory, subclass");
    assert(make(Shape, 3).area() == 0, "factory, the class itself");
    assert(make(Square, 3).name == "square" && make(Triple, 3).name == "shape", "constructors and initializers");

    const kinds: (typeof Shape)[] = [Square, Triple, Shape];
    let total: number = 0;
    for (const k of kinds) {
        total += new k(2).area();
    }

    assert(total == 4 + 12, "array of classes");

    let c = pick(1);
    assert(new c(1).area() == 3, "returned");
    c = pick(0);
    assert(new c(5).area() == 25, "assigned");

    const h = new Holder(Square);
    assert(new h.kind(4).area() == 16, "field");

    const p: typeof Plain = Plain;
    assert(new p().v == 5, "no constructor");

    assert(new (pick(0))(4) instanceof Square, "instanceof the instance");

    const a: typeof Shape = Square;
    const b: typeof Shape = Square;
    const t: typeof Shape = Triple;
    assert(a == b && a != t && !(a === t), "equality");

    const x: typeof Base = Impl;
    const y: typeof Base = Base;
    assert(x != y && y == Base, "abstract classes");

    assert(typeof a == "function", "typeof");

    assert(countOf(Counter) == 7, "static through the value");

    assert(labelOf(Double) == "double" && labelOf(Unit) == "unit" && labelOf(Inherits) == "unit", "static methods");
    const units: (typeof Unit)[] = [Double, Inherits];
    let scaled: number = 0;
    for (const u of units) {
        scaled += u.scaled(3);
    }

    assert(scaled == 9, "static method with arguments");

    const kind: typeof Kind = Kind;
    const concrete: typeof Kind = Concrete;
    assert(kind.id() == 1 && kind.n == 4 && concrete.id() == 2 && concrete.n == 4, "statics of an abstract class's value");
    assert(new Concrete().title() == "concrete", "abstract slot filled");

    print("done.");
}

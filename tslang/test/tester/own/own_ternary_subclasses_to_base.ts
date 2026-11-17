// `c ? new Sq(..) : new Ci(..)` given to a `Shape`: the two subclasses make a union, cast to the
// base. That cast called a generated `___cast`, whose result -mm=own took for a borrow of the union,
// so a function could not return it and a field could not keep it. Now it is an upcast of the
// pointer the union holds. Values are made at run time, in a loop, with strings that own memory, so
// a double release or a leak shows.
class Shape {
    constructor(public name: string) {}
    area(): number {
        return 0;
    }
}

class Sq extends Shape {
    constructor(public s: number) {
        super("sq" + s);
    }
    area(): number {
        return this.s * this.s;
    }
}

class Ci extends Shape {
    constructor(public r: number) {
        super("ci" + r);
    }
    area(): number {
        return 3 * this.r * this.r;
    }
}

class Holder {
    shape: Shape = new Shape("none");
}

function make(i: number): Shape {
    return i % 2 == 0 ? new Sq(i) : new Ci(i);
}

function makeConst(i: number): Shape {
    const s: Shape = i % 2 == 0 ? new Ci(i) : new Sq(i);
    return s;
}

function main() {
    const h = new Holder();
    let total: number = 0;
    for (let i = 0; i < 2000; i++) {
        const a = make(i);
        assert(a.area() == (i % 2 == 0 ? i * i : 3 * i * i), "returned: the right subclass's method");
        assert(a.name == (i % 2 == 0 ? "sq" : "ci") + i, "returned: its fields");

        const b = makeConst(i);
        assert(b.name == (i % 2 == 0 ? "ci" : "sq") + i, "through a const");

        h.shape = i % 3 == 0 ? new Sq(i) : new Ci(i);
        assert(h.shape.name == (i % 3 == 0 ? "sq" : "ci") + i, "kept by a field");

        total += a.area() + b.area() + h.shape.area();
    }

    assert(total > 0, "total");
    print("done.");
}

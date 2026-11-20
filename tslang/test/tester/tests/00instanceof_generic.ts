// `x instanceof Box` with a generic class: every instance of it (`Box<number>`, `Box<string>`, a class
// extending one) is a Box. Each instance class answers its generic's name besides its own rtti
// (mlirGenClassInstanceOfMethod), and the generic is asked by its name.
class Box<T> {
    constructor(public v: T) {}
}

class Other {}

class Sub extends Box<number> {}

namespace NS {
    export class Pair<A, B> {
        constructor(public a: A, public b: B) {}
    }
}

function isBox(x: any) {
    return x instanceof Box;
}

function main() {
    const b = new Box<number>(1);
    const s = new Box<string>("a");
    const o = new Other();
    assert(b instanceof Box, "Box<number>");
    assert(s instanceof Box, "Box<string>");
    assert(!(o instanceof Box), "another class");

    const u: Box<number> | Other = new Box<number>(2);
    const u2: Box<number> | Other = new Other();
    assert(u instanceof Box, "a union holding a Box");
    assert(!(u2 instanceof Box), "a union holding another class");

    const sub = new Sub(3);
    assert(sub instanceof Box, "a class extending Box<number>");
    assert(sub instanceof Sub, "the class itself");
    assert(!(b instanceof Sub), "a base is not its subclass");

    assert(isBox(new Box<number>(4)), "an any holding a Box");
    assert(!isBox(new Other()), "an any holding another class");
    assert(isBox(new Box<Box<number>>(new Box<number>(5))), "a Box of a Box");

    const p = new NS.Pair<number, (x: number) => string>(1, (x: number) => "y");
    assert(p instanceof NS.Pair, "a generic in a namespace, a function type argument");

    print("done.");
}

// TSLANG 171: a generic class given to a constructor type with no type arguments (`make(Box, 4)`) takes them from
// the constructor type's parameters, as `new Box(4)` takes them from its arguments

class Box<T> {
    constructor(public v: T) {}
}

class Pair<A, B> {
    constructor(public a: A, public b: B) {}
}

function make<T>(ctor: new (n: number) => T, n: number): T {
    return new ctor(n);
}

function makePair<T>(ctor: new (s: string, n: number) => T): T {
    return new ctor("p", 2.5);
}

function main() {
    const b = make(Box, 4);
    assert(b.v == 4);

    const c = make(Box<number>, 3);
    assert(c.v == 3);

    const k = Box<number>;
    assert(make(k, 5).v == 5);

    const p = makePair(Pair);
    assert(p.a == "p");
    assert(p.b == 2.5);

    let total = 0;
    for (let i = 0; i < 1000; i++) total += make(Box, i).v;
    assert(total == 499500);

    print("done.");
}
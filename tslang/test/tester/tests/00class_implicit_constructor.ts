// regression test: a derived class without a constructor gets TypeScript's implicit one,
// `constructor(...args) { super(...args); }`.
//
// bugs found and fixed:
// - a derived class with field initializers (and no constructor) was given a generated
//   `constructor() { super(); }`, so `new Derived(2, 3)` failed with "Expected 1-2 arguments,
//   but got 0", reported at line 1. It now takes the base constructor's parameters (optional
//   ones and a rest parameter included) and passes them on.
// - `super(a, ...rest)` passed an empty rest parameter: the arguments of a super call were
//   generated without the constructor's parameters, so a spread had nothing to fill.

class Base {
    constructor(protected a: int, protected b: int = 10, public c?: string, ...rest: int[]) {
        this.restLength = rest.length;
    }

    restLength: int;

    sum(): int {
        return this.a + this.b;
    }
}

class Derived extends Base {
    extra = 2;
}

// the implicit constructor of a class whose base has one too
class MoreDerived extends Derived {
    more = 3;
}

class SpreadSuper extends Base {
    constructor(a: int, ...rest: int[]) {
        super(a, 1, "s", ...rest);
    }
}

class Generic<T> {
    constructor(public v: T) {
    }
}

class FromSpecialized extends Generic<string> {
    w = 5;
}

class FromGeneric<T> extends Generic<T> {
    w = 6;
}

class NoConstructorBase {
    x = 1;
}

class NoConstructorDerived extends NoConstructorBase {
    y = 2;
}

// a field initializer runs in the implicit constructor, and still sees the outer `arg0`
const arg0 = 99;
class Shadow extends Base {
    z = arg0;
}

function main() {
    const d = new Derived(2, 3, "s", 7, 8);
    assert(d.sum() == 5);
    assert(d.extra == 2);
    assert(d.c == "s");
    assert(d.restLength == 2);

    // the base's default and optional parameters
    const d2 = new Derived(4);
    assert(d2.sum() == 14);
    assert(d2.c == undefined);
    assert(d2.restLength == 0);

    const m = new MoreDerived(1, 2, "t", 9);
    assert(m.sum() == 3);
    assert(m.extra == 2);
    assert(m.more == 3);
    assert(m.restLength == 1);

    const s = new SpreadSuper(1, 7, 8, 9);
    assert(s.restLength == 3);

    const f = new FromSpecialized("hello");
    assert(f.v == "hello");
    assert(f.w == 5);

    const g = new FromGeneric<number>(1.5);
    assert(g.v == 1.5);
    assert(g.w == 6);

    const n = new NoConstructorDerived();
    assert(n.x == 1);
    assert(n.y == 2);

    assert(new Shadow(1).z == 99);

    print("done.");
}

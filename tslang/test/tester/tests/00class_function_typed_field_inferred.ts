// regression test: a class field whose function type is inferred from its
// initializer.
// - `fn = () => 0` was stored as a plain function, which has no room for a
//   closure's context: assigning a capturing closure to it warned "losing this
//   reference" and the call crashed. It is stored as a hybrid function now, as
//   an annotated `fn: () => int` field and a `let` are.
// - `fn = this.m` had no type in the first pass over the program (the class's
//   methods aren't known while its fields are), so the field was `any` there
//   and `w.fn()` failed with "not a function to call".

interface HasFn {
    fn: () => int;
}

class Capturing {
    fn = () => 0;
    constructor(k: int) {
        this.fn = () => k;
    }
}

class WithParams {
    add = (a: int, b: int) => a + b;
}

class BoundMethod {
    v = 4;
    m(): int {
        return this.v;
    }
    fn = this.m;
}

class BoundBeforeMethod {
    fn = this.twice;
    constructor(public v: int) {}
    twice(): int {
        return this.v * 2;
    }
}

function main() {
    const c = new Capturing(9);
    print(c.fn());
    assert(c.fn() == 9);

    let k2 = 5;
    c.fn = () => k2 + 1;
    assert(c.fn() == 6);
    k2 = 10;
    assert(c.fn() == 11);

    const p = new WithParams();
    assert(p.add(2, 3) == 5);
    const base = 100;
    p.add = (a: int, b: int) => a * b + base;
    assert(p.add(2, 3) == 106);

    // under -mm=rc a bound method does not hold its receiver yet: `b` keeps it alive
    const b = new BoundMethod();
    print(b.fn());
    assert(b.fn() == 4);
    b.v = 7;
    assert(b.fn() == 7);
    const g = b.fn;
    assert(g() == 7);

    const bb = new BoundBeforeMethod(21);
    assert(bb.fn() == 42);

    // an inferred function-typed field fills an interface's function-typed field
    const h: HasFn = c;
    assert(h.fn() == 11);
    const hb: HasFn = b;
    assert(hb.fn() == 7);

    // a closure replaces the bound method
    const k3 = 3;
    b.fn = () => k3;
    assert(b.fn() == 3);

    print("done.");
}

// regression test: a class method fills an interface member declared as a
// function-typed FIELD (`fn: () => int`). The class has no field holding the
// method, so the cast to the interface used to fail ("field type not matching
// for "fn""), which then surfaced as "not a function to call" at the caller.
// The slot now points at a cell holding the method (INTERFACE_METHOD_CELL_TAG,
// Defines.h), and reading the field binds `this` as for any function-typed field.

interface HasFn {
    fn: () => int;
}

class C {
    constructor(public v: int) {}
    fn(): int {
        return this.v;
    }
}

// under -mm=rc a bound method does not hold its receiver yet, so the caller keeps
// the instance alive while the returned value is used
function fromClass(c: C): () => int {
    const o: HasFn = c;
    return o.fn;
}

interface Adder {
    base: int;
    add: (x: int) => int;
}

class A {
    constructor(public base: int) {}
    add(x: int): int {
        return this.base + x;
    }
}

class Base {
    constructor(public v: int) {}
    fn(): int {
        return this.v * 2;
    }
}

class Derived extends Base {
    constructor(v: int) {
        super(v);
    }
}

function callFn(h: HasFn): int {
    return h.fn();
}

// declared with `implements`, inherited from a base that declares it, and reached twice
// through a diamond `extends` (two slots, one cell)
interface P1 extends HasFn {
    a: int;
}

interface P2 extends HasFn {
    b: int;
}

interface Both extends P1, P2 {}

class Impl implements HasFn {
    constructor(public v: int) {}
    fn(): int {
        return this.v + 1;
    }
}

class ImplBase implements HasFn {
    constructor(public v: int) {}
    fn(): int {
        return this.v * 3;
    }
}

class ImplDerived extends ImplBase {
    constructor(v: int) {
        super(v);
    }
}

class Diamond {
    a = 1;
    b = 2;
    constructor(public v: int) {}
    fn(): int {
        return this.v + 100;
    }
}

function main() {
    const c42 = new C(42);
    const f = fromClass(c42);
    print(f());
    assert(f() == 42);

    const o: HasFn = new C(5);
    print(o.fn());
    assert(o.fn() == 5);

    const a: Adder = new A(10);
    assert(a.base == 10);
    assert(a.add(3) == 13);
    const add = a.add;
    assert(add(4) == 14);

    const d: HasFn = new Derived(21);
    assert(d.fn() == 42);

    // one access site: a class's method cell, an object literal's field
    assert(callFn(new C(1)) == 1);
    assert(callFn({ v: 41, fn() { return this.v; } }) == 41);

    const i: HasFn = new Impl(6);
    assert(i.fn() == 7);
    const id: HasFn = new ImplDerived(4);
    assert(id.fn() == 12);
    const both: Both = new Diamond(5);
    assert(both.fn() == 105);
    assert(both.a == 1 && both.b == 2);

    print("done.");
}

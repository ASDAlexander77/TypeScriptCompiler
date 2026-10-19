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

function fromClass(): () => int {
    const o: HasFn = new C(42);
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

function main() {
    const f = fromClass();
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

    print("done.");
}

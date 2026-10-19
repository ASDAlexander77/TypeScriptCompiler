// regression test: a class FIELD holding a function fills an interface member
// declared as a function-typed field (`fn: () => int`). The class stores the
// field as a hybrid function ({function, `this` or null}), which the interface
// slot can't hand out as a function taking the object as its `this`, so the
// cast used to fail ("field type not matching for "fn""). The slot now points
// at a cell holding a thunk that calls `this.fn(...)` (INTERFACE_METHOD_CELL_TAG,
// Defines.h), so the field is read at every call.

interface HasFn {
    fn: () => int;
}

class WithField {
    fn: () => int = () => 7;
}

// unannotated: stored as a plain function
class Unannotated {
    fn = () => 6;
}

class Capturing {
    fn: () => int;
    constructor(public x: int, k: int) {
        this.fn = () => this.x + k;
    }
}

class BoundMethod {
    fn: () => int;
    constructor(public v: int) {
        this.fn = this.m;
    }
    m(): int {
        return this.v * 2;
    }
}

interface Fmt {
    fmt: (a: int, s: string) => string;
}

class FmtField {
    constructor(public prefix: string) {}
    fmt: (a: int, s: string) => string = (a: int, s: string) => `${s}=${a}`;
}

let calls = 0;

interface Sink {
    put: (v: int) => void;
}

class SinkField {
    put: (v: int) => void = (v: int) => {
        calls += v;
    };
}

class Implements implements HasFn {
    fn: () => int = () => 11;
}

class FieldBase {
    fn: () => int = () => 12;
}

class FieldDerived extends FieldBase {}

interface P1 extends HasFn {
    a: int;
}

interface P2 extends HasFn {
    b: int;
}

interface Both extends P1, P2 {}

class Diamond {
    a = 1;
    b = 2;
    fn: () => int = () => 13;
}

function callFn(h: HasFn): int {
    return h.fn();
}

// under -mm=rc a bound method does not hold its receiver yet, so the caller keeps
// the instance alive while the returned value is used
function fromField(w: WithField): () => int {
    const h: HasFn = w;
    return h.fn;
}

function main() {
    const h: HasFn = new WithField();
    print(h.fn());
    assert(h.fn() == 7);

    const u: HasFn = new Unannotated();
    assert(u.fn() == 6);

    const c: HasFn = new Capturing(30, 5);
    assert(c.fn() == 35);

    const b: HasFn = new BoundMethod(21);
    assert(b.fn() == 42);

    // the thunk reads the field at each call
    const w = new WithField();
    const hw: HasFn = w;
    w.fn = () => 8;
    assert(hw.fn() == 8);

    const f: Fmt = new FmtField("p");
    print(f.fmt(3, "x"));
    assert(f.fmt(3, "x") == "x=3");

    const s: Sink = new SinkField();
    s.put(4);
    s.put(5);
    assert(calls == 9);

    const i: HasFn = new Implements();
    assert(i.fn() == 11);
    const d: HasFn = new FieldDerived();
    assert(d.fn() == 12);
    const both: Both = new Diamond();
    assert(both.fn() == 13);
    assert(both.a == 1 && both.b == 2);

    // one access site: a class field, a class with a method, an object literal
    assert(callFn(new WithField()) == 7);
    assert(callFn({ fn() { return 41; } }) == 41);

    const wf = new WithField();
    const g = fromField(wf);
    assert(g() == 7);

    print("done.");
}

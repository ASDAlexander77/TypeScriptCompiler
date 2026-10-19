// regression test: a class's OPTIONAL function-typed field (`fn?: () => int`)
// filling an interface's optional function-typed field. The interface slot
// points at a method cell shared by every instance of the class, holding a
// thunk that calls `this.fn(...)` (INTERFACE_METHOD_CELL_TAG, Defines.h); the
// cell also carries the offset of the field's presence flag, so each instance
// reads as present only while its own field is set. It was a compile error
// ("field type not matching"): without the flag an unset field read as present
// and the thunk called it.

interface MaybeFn {
    fn?: () => int;
}

function maybe(m: MaybeFn): int {
    const g = m.fn;
    if (g) {
        return g();
    }

    return -1;
}

function callOpt(m: MaybeFn): int {
    return m.fn?.() ?? -2;
}

function isSet(m: MaybeFn): boolean {
    return m.fn !== undefined;
}

class OptField {
    fn?: () => int;
    constructor(public v: int, set: boolean) {
        if (set) {
            this.fn = () => this.v;
        }
    }
}

// the flag sits after a field of another size
class OptFieldAfter {
    a = 1;
    fn?: () => int;
    b = 2;
}

class WithFn {
    constructor(public v: int) {}
    fn(): int {
        return this.v;
    }
}

class WithoutFn {
    constructor(public v: int) {}
}

interface MaybeAdd {
    base: int;
    add?: (x: int) => int;
}

function addOr(m: MaybeAdd, x: int): int {
    const add = m.add;
    if (add) {
        return add(x);
    }

    return m.base;
}

class OptAdder {
    add?: (x: int) => int;
    constructor(public base: int) {}
}

// the field is inherited
class OptFieldDerived extends OptField {
    constructor(v: int, set: boolean) {
        super(v, set);
    }
}

function main() {
    // set, unset; under -mm=rc a bound value does not hold its receiver yet, so each
    // instance is kept in a local while it is used through the interface
    const s21 = new OptField(21, true);
    const u22 = new OptField(22, false);
    print(maybe(s21), maybe(u22));
    assert(maybe(s21) == 21);
    assert(maybe(u22) == -1);
    assert(callOpt(s21) == 21);
    assert(callOpt(u22) == -2);
    assert(isSet(s21));
    assert(!isSet(u22));

    // assigned later, through the class
    const o = new OptField(1, false);
    const m: MaybeFn = o;
    print(maybe(m));
    assert(maybe(m) == -1);
    o.fn = () => 9;
    print(maybe(m));
    assert(maybe(m) == 9);
    print(m.fn?.() ?? -2);
    assert((m.fn?.() ?? -2) == 9);
    assert(isSet(m));

    // assigned undefined again
    o.fn = undefined;
    assert(maybe(m) == -1);
    assert(callOpt(m) == -2);
    assert(!isSet(m));

    // the cell is shared: setting one instance's field leaves another unset
    const p = new OptField(2, false);
    const q = new OptField(3, false);
    p.fn = () => 30;
    assert(maybe(p) == 30);
    assert(maybe(q) == -1);
    q.fn = () => p.v + q.v;
    assert(maybe(p) == 30);
    assert(maybe(q) == 5);

    // the field after another one
    const after = new OptFieldAfter();
    assert(maybe(after) == -1);
    after.fn = () => after.a + after.b;
    assert(maybe(after) == 3);
    assert(after.a == 1 && after.b == 2);

    // with a parameter
    const adder = new OptAdder(10);
    assert(addOr(adder, 5) == 10);
    adder.add = (x: int) => adder.base + x;
    assert(addOr(adder, 5) == 15);

    // inherited
    const d1 = new OptFieldDerived(41, true);
    const d2 = new OptFieldDerived(42, false);
    assert(maybe(d1) == 41);
    assert(maybe(d2) == -1);

    // one access site, mixed implementers
    const set4 = new OptField(4, true);
    const unset5 = new OptField(5, false);
    const with6 = new WithFn(6);
    const without7 = new WithoutFn(7);
    const all: MaybeFn[] = [set4, unset5, with6, without7, { fn() { return 8; } }, {}];
    let sum = 0;
    for (const a of all) {
        sum += maybe(a);
    }

    print(sum);
    assert(sum == 4 - 1 + 6 - 1 + 8 - 1);

    unset5.fn = () => 50;
    set4.fn = undefined;
    sum = 0;
    for (const a of all) {
        sum += callOpt(a);
    }

    assert(sum == -2 + 50 + 6 - 2 + 8 - 2);

    print("done.");
}

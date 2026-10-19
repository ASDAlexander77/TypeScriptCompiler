// regression test: reading an OPTIONAL function-typed interface field
// (`fn?: () => int`) as a value. The field loads as optional<function>, and
// binding the interface's `this` to it directly failed to compile
// ('ts.CreateBoundFunction' op operand #1 must be ...). A present field now
// reads as an optional bound function, an absent one as undefined - for an
// object literal and for a class whose method fills the field
// (INTERFACE_METHOD_CELL_TAG, Defines.h) alike.
//
// A class's own optional function-typed field (`fn?: () => int`) filling the
// interface's is a compile error for now: interface-errors/optional_class_function_field.ts.

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

function getFn(m: MaybeFn): (() => int) | undefined {
    return m.fn;
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

class Adder {
    constructor(public base: int) {}
    add(x: int): int {
        return this.base + x;
    }
}

function main() {
    // object literal
    print(maybe({ fn() { return 3; } }), maybe({}));
    assert(maybe({ fn() { return 3; } }) == 3);
    assert(maybe({}) == -1);
    assert(callOpt({ fn() { return 4; } }) == 4);
    assert(callOpt({}) == -2);
    assert(isSet({ fn() { return 5; } }));
    assert(!isSet({}));

    // the bound value keeps the literal's `this`
    assert(maybe({ v: 40, fn() { return this.v + 2; } }) == 42);

    // a class whose method fills the field, and one that lacks it
    assert(maybe(new WithFn(7)) == 7);
    assert(maybe(new WithoutFn(8)) == -1);
    assert(callOpt(new WithFn(9)) == 9);
    assert(callOpt(new WithoutFn(10)) == -2);
    assert(isSet(new WithFn(11)));
    assert(!isSet(new WithoutFn(12)));

    // one access site, mixed implementers
    const all: MaybeFn[] = [new WithFn(1), new WithoutFn(2), { fn() { return 3; } }, {}];
    let sum = 0;
    for (const m of all) {
        sum += maybe(m);
    }

    print(sum);
    assert(sum == 1 - 1 + 3 - 1);

    // the bound value outlives the access; under -mm=rc a bound method does not hold its
    // receiver yet, so the instance is kept alive while the returned value is used
    const w13 = new WithFn(13);
    const f = getFn(w13);
    assert(f !== undefined);
    if (f) {
        assert(f() == 13);
    }

    assert(getFn(new WithoutFn(14)) === undefined);

    // with a parameter
    assert(addOr(new Adder(10), 5) == 15);
    assert(addOr({ base: 20, add(x: int) { return this.base * x; } }, 2) == 40);
    assert(addOr({ base: 30 }, 2) == 30);

    print("done.");
}

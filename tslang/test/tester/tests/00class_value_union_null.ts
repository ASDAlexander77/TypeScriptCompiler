// Class values and class instances with `| null` / `| undefined`:
// - `typeof A | null` was stored as `typeof A` (null dropped, as for a type that holds null in its own
//   pointer; a class value is four pointers), so assigning null was an invalid cast;
// - `new c()` of such a value built a `ts.New` of the union type (invalid IR) - it makes the class the
//   value holds now, and throws when it holds none;
// - a class value as a condition (`if (k)`) was an invalid cast, and of a union of them "not supported";
// - `o.name()` of `A | B | null` called A's method whatever `o` held.
class A {
    constructor(public n = 0) {}
    name() {
        return "A";
    }
}

class B {
    constructor(public n = 0) {}
    name() {
        return "B";
    }
}

function make(c: typeof A | typeof B | null, n: number) {
    return new c(n);
}

function makeOne(c: typeof A | undefined, n: number) {
    return new c(n);
}

function isSet(c: typeof A | null) {
    return !!c;
}

function isAnySet(c: typeof A | typeof B | null) {
    if (c) {
        return true;
    }

    return false;
}

function main() {
    // `typeof A | null` holds null, and a class
    let c: typeof A | null = null;
    assert(c == null && !isSet(c), "null");
    c = A;
    assert(c != null && isSet(c), "a class");
    if (c) {
        assert(new c(4).n == 4, "made");
    }

    // a union of class values with null: made, each its class
    const b = make(B, 2);
    assert(b instanceof B && !(b instanceof A) && b.n == 2, "make(B)");
    const a = make(A, 3);
    assert(a instanceof A && a.n == 3, "make(A)");
    assert(makeOne(A, 5).n == 5, "makeOne(A)");

    // none to make: it throws
    let thrown = 0;
    try {
        make(null, 1);
    } catch (e) {
        thrown++;
    }

    try {
        makeOne(undefined, 1);
    } catch (e) {
        thrown++;
    }

    assert(thrown == 2, "no class: throws");

    // as conditions
    assert(!isAnySet(null), "union null");
    assert(isAnySet(B), "union class");
    const only: typeof A = A;
    let truthy = false;
    if (only) {
        truthy = true;
    }

    assert(truthy, "a class value is true");

    // a method of `A | B | null`: the class the object is of
    const o: A | B | null = new B();
    assert(o.name() == "B", "dispatch with null");
    let p: A | B | undefined = new A();
    if (p) {
        assert(p.name() == "A", "dispatch with undefined");
    }

    print("done.");
}

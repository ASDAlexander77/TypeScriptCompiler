// TSLANG_ISSUES 46 (-mm=own): a function returning its union parameter, given a new object, freed the
// object twice. `same(new A())` to an `A | null` parameter is given `<A | null>new A()`: the result
// was made the caller's own while the object's release stayed. The releases taken over are the
// object's now, after the call in its block or in one only the call leads to. And `x instanceof A`
// of a borrowed union (the result of such a call, or a field read) is a test that keeps nothing.

class A {
    v = 1;
}

class B { }

function same(v: A | null): A | null {
    return v;
}

function sameTagged(v: boolean | A): boolean | A {
    return v;
}

function widen(v: null | boolean | A): null | boolean | A | B {
    return v;
}

function make(): A {
    return new A();
}

class Holder {
    f: A | null = new A();

    has(): boolean {
        const x = this.f;
        return x instanceof A;
    }
}

function main() {
    // a new object given straight to the call
    const x = same(new A());
    assert(x instanceof A, "same(new A())");

    const t = sameTagged(new A());
    assert(t instanceof A, "sameTagged(new A())");

    const w = widen(new A());
    assert(w instanceof A, "widen(new A())");
    assert(widen(true) === true, "widen(true)");
    assert(widen(null) === null, "widen(null)");

    // a named object, used after the call: the result borrows it
    const a = new A();
    const y = same(a);
    assert(a instanceof A && y instanceof A && a.v == 1, "same(a), a used after");

    // what a call returns, given on
    const z = same(make());
    assert(z instanceof A, "same(make())");

    // in a loop
    let sum = 0;
    for (let i = 0; i < 3; i++) {
        const e = same(new A());
        if (e) {
            sum += e.v;
        }
    }

    assert(sum == 3, "same(new A()) in a loop");

    // a field read tested with instanceof
    assert(new Holder().has(), "instanceof of a borrowed field");

    print("done.");
}

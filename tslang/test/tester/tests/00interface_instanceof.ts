// An interface value can be asked what it holds: slot 0 of every interface vtable is the
// implementer's `.instanceOf`, or, for an object literal, one that answers false.
//
// Before, `i instanceof B` on an interface was decided at compile time and was always false;
// `<B>i`, `i as B` and narrowing by `instanceof` crashed the compiler (it tried to build a new B
// out of the interface's fields); and `<B>a`, with `a` an `any` holding an interface, threw.

interface I {
    get(): number;
}

interface J {
    twice(): number;
}

class B implements I, J {
    x: number = 5;
    get() {
        return this.x;
    }

    twice() {
        return this.x * 2;
    }
}

class D extends B {
    y: number = 1;
}

class Other implements I {
    get(): number {
        return 9;
    }
}

function main() {
    const i: I = new B();
    const j: J = new B();
    const d: I = new D();
    const lit: I = { get(): number { return 3; } };

    assert(i instanceof B);
    assert(!(i instanceof Other));
    assert(!(i instanceof D));
    assert(j instanceof B);
    assert(d instanceof B);
    assert(d instanceof D);
    assert(!(lit instanceof B));

    // methods and fields still reach their slots, after the one in front of them
    assert(i.get() == 5);
    assert(j.twice() == 10);
    assert(lit.get() == 3);

    // a downcast is the very object
    const b = <B>i;
    b.x = 6;
    assert(i.get() == 6);
    assert((i as B).twice() == 12);
    if (d instanceof D) {
        assert(d.y == 1);
    }

    // a class with no field outside the interface is still converted from what is not one
    assert((i as Other).get() == 9);

    // an `any` holding an interface
    const a: any = i;
    assert((<B>a).x == 6);
    const al: any = lit;
    let caught = false;
    try {
        const nb = <B>al;
        print(nb.x);
    } catch (e) {
        caught = true;
    }

    assert(caught);

    print("done.");
}

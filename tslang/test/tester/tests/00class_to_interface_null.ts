// @strict-null false
// A null interface is an interface over a null `this`, seen through the interface's null vtable,
// whose methods throw, as a call of an abstract method would. A null class object given to an
// interface is one, and so are `null` and `undefined`. An interface is tested for null, and
// compared, by its `this`.
//
// A null class object was paired with its class's vtable, so the interface was not `=== null`; for a
// class naming the interface, whose vtable is read through the object, the cast read through the
// null pointer and crashed.
interface I {
    x: number;
    get(): number;
}

class C {
    constructor(public x: number) {}

    get(): number {
        return this.x * 10;
    }
}

class D implements I {
    constructor(public x: number) {}

    get(): number {
        return this.x * 100;
    }
}

@noinline function mkC(flag: boolean, x: number): C {
    return flag ? new C(x) : null;
}

@noinline function mkD(flag: boolean, x: number): D {
    return flag ? new D(x) : null;
}

function read(i: I): number {
    if (i === null) {
        return -1;
    }

    return i.x + i.get();
}

// what a method of the interface does: -2 when it throws an Error
function call(i: I): number {
    try {
        return i.get();
    } catch (e) {
        return e instanceof Error ? -2 : -3;
    }
}

function main() {
    // a field first, and no method: a method called through a wrong vtable runs whatever code it lands on
    const first: I = mkC(true, 1);
    assert(first !== null && first.x == 1, "a field read first");

    const nc: I = mkC(false, 2);
    assert(nc === null, "a null class object");
    assert(nc == null, "a null class object, loosely");
    assert(!nc, "a null class object is false");
    const nd: I = mkD(false, 3);
    assert(nd === null, "a null object of a class naming the interface");
    const nl: I = null;
    assert(nl === null, "null");
    const nu: I = undefined;
    assert(nu == null, "undefined is loosely null");
    assert(nu === undefined, "undefined");

    assert(call(nc) == -2, "a method of a null class object throws");
    assert(call(nd) == -2, "a method of a null object of a class naming the interface throws");
    assert(call(nl) == -2, "a method of null throws");
    assert(call(nu) == -2, "a method of undefined throws");
    assert(call(mkC(true, 5)) == 50, "a method of a class object");

    assert(read(mkC(true, 4)) == 44, "an argument");
    assert(read(mkC(false, 4)) == -1, "a null argument");
    assert(read(mkD(true, 5)) == 505, "an argument of a class naming the interface");
    assert(read(mkD(false, 5)) == -1, "a null argument of a class naming the interface");

    // compared by the object
    const c = new C(6);
    const i1: I = c;
    const i2: I = mkC(true, 6);
    assert(i1 !== i2, "two objects");
    assert(i1 === i1, "the same object");

    print("done.");
}

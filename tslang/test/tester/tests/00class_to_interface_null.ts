// @strict-null false
// A null class object given to an interface is a null interface. A class object is a pointer that
// may be null; seen through an interface it was paired with its class's vtable all the same, so
// the interface was not `=== null` - or, for a class naming the interface, whose vtable is read
// through the object, the cast read through the null pointer and crashed.
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

function main() {
    // a field first, and no method: a method called through a wrong vtable runs whatever code it lands on
    const first: I = mkC(true, 1);
    assert(first !== null && first.x == 1, "a field read first");

    const nc: I = mkC(false, 2);
    assert(nc === null, "a null class object");
    const nd: I = mkD(false, 3);
    assert(nd === null, "a null object of a class naming the interface");

    assert(read(mkC(true, 4)) == 44, "an argument");
    assert(read(mkC(false, 4)) == -1, "a null argument");
    assert(read(mkD(true, 5)) == 505, "an argument of a class naming the interface");
    assert(read(mkD(false, 5)) == -1, "a null argument of a class naming the interface");

    print("done.");
}

// A class object or null (`C | null`) given to an interface or null (`I | null`) becomes an
// interface - the class object paired with the vtable it is seen through - and null stays null.
// The conversion was a plain cast that read the class pointer as an interface, under every model:
// the lowering could not make it, and the program crashed or read garbage. A class object alone
// (`let a: I = mk(5)`) was already converted; a value of the class type with null beside it was
// not. The `held` shapes read the interface after the class-typed local let go of the object and
// allocations of the same size reused anything freed.
interface I {
    x: number;
    get(): number;
}

class C {
    pad: number = 7;
    constructor(public x: number) {}

    get(): number {
        return this.x * 10;
    }
}

// a class that names the interface: its vtable is read through the object
class D implements I {
    constructor(public x: number) {}

    get(): number {
        return this.x * 100;
    }
}

@noinline function maybeD(flag: boolean, x: number): D | null {
    return flag ? new D(x) : null;
}

@noinline function mk(x: number): C {
    return new C(x);
}

@noinline function maybe(flag: boolean, x: number): C | null {
    return flag ? mk(x) : null;
}

function read(i: I | null): number {
    if (i === null) {
        return -1;
    }

    return i.x + i.get();
}

function viewed(c: C | null): I | null {
    return c;
}

function conditional(flag: boolean) {
    let opt: I | null = flag ? mk(5) : null;
    if (opt === null) {
        return -1;
    }

    return opt.x + opt.get();
}

function constConditional(flag: boolean) {
    const opt: I | null = flag ? mk(6) : null;
    return opt !== null ? opt.x : -1;
}

function assigned(flag: boolean) {
    let opt: I | null = null;
    const c = maybe(flag, 8);
    opt = c;
    return read(opt);
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new C(777));
    }
}

function held(flag: boolean) {
    let c: C | null = maybe(flag, 11);
    const opt: I | null = c;
    c = null;
    churn();
    return read(opt);
}

let G: I | null = maybe(true, 12);
let H: I | null = maybe(false, 12);

function main() {
    // A field first, and no method: a method called through a wrong vtable runs whatever code it
    // lands on, which allocated without end (gigabytes) when the conversion was a plain cast.
    const first: I | null = maybe(true, 1);
    assert(first !== null && first.x == 1, "a field read first");

    assert(held(true) == 121, "held after the class-typed local let go");
    assert(held(false) == -1, "a held null");

    churn();
    assert(read(G) == 132, "a global");
    assert(H === null, "a null global");
    G = maybe(true, 13);
    churn();
    assert(read(G) == 143, "a global reassigned");

    assert(conditional(true) == 55, "a conditional, class side");
    assert(conditional(false) == -1, "a conditional, null side");

    assert(constConditional(true) == 6, "a const, class side");
    assert(constConditional(false) == -1, "a const, null side");

    assert(read(maybe(true, 3)) == 33, "an argument");
    assert(read(maybe(false, 3)) == -1, "a null argument");

    const v = viewed(maybe(true, 4));
    assert(v !== null && v.x == 4 && v.get() == 40, "a returned value");
    assert(viewed(maybe(false, 4)) === null, "a returned null");

    assert(assigned(true) == 88, "an assignment");
    assert(assigned(false) == -1, "a null assignment");

    assert(read(maybeD(true, 2)) == 202, "a class implementing the interface");
    assert(read(maybeD(false, 2)) == -1, "a null of a class implementing the interface");
    const d: I | null = maybeD(true, 3);
    assert(d !== null && d.get() == 300, "a const of a class implementing the interface");

    let u: I | undefined = undefined;
    const cu: C | undefined = mk(9);
    u = cu;
    assert(u !== undefined && u.x == 9 && u.get() == 90, "an optional");

    print("done.");
}

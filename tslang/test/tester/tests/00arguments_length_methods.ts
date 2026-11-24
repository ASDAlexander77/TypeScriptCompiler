// TSLANG 187: `arguments.length` counts an `undefined` given in a class's methods, static methods and constructors, and
// in generic functions, as in plain functions (186): a call by the method's symbol - direct, through the vtable or
// static - gives the count. An override with or without `arguments`, or with fewer parameters, works through the
// base class's slot.
class A {
    m(a: number, b?: number): number {
        return arguments.length;
    }

    static s(a: number, b?: number): number {
        return arguments.length;
    }

    n(a: number, b?: number): number {
        return this.m(a, undefined) * 10 + arguments.length;
    }

    k(a: number): string {
        return "A";
    }
}

class B extends A {
    m(a: number, b?: number): number {
        return 100 + arguments.length;
    }

    k(a: number, b?: string): string {
        return "B" + arguments.length;
    }
}

class C extends A {
    m(a: number): number {
        return 200 + a;
    }
}

class P {
    n: number;
    constructor(a: number, b?: number) {
        this.n = arguments.length;
    }
}

function g<T>(a: T, b?: T): number {
    return arguments.length;
}

function main() {
    const a = new A();
    assert(a.m(1) == 1 && a.m(1, undefined) == 2 && a.m(1, 2) == 2, "method");
    assert(A.s(1) == 1 && A.s(1, undefined) == 2, "static");
    assert(a.n(1) == 21, "this.m");

    const x: A = new B();
    assert(x.m(1) == 101 && x.m(1, undefined) == 102, "override through the base");
    assert(x.k(1) == "B1", "an override reading arguments, called through a base that does not");
    const y = new B();
    assert(y.k(1) == "B1" && y.k(1, undefined) == "B2", "override");
    const z: A = new C();
    assert(z.m(5) == 205 && z.m(5, undefined) == 205, "an override with fewer parameters");

    assert(new P(1).n == 1 && new P(1, undefined).n == 2 && new P(1, 2).n == 2, "constructor");
    assert(g(1) == 1 && g(1, undefined) == 2 && g("s", "t") == 2, "generic");

    print("done.");
}

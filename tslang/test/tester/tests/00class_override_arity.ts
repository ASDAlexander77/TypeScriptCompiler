// overrides of another arity than the method they override, called through the base class and
// through their own class, inside and outside the class bodies

abstract class Base {
    abstract get(): number;

    twice() {
        return this.get() * 2;
    }
}

class Impl extends Base {
    get(): number;
    get(scale: number): number;
    get(scale?: number) {
        return scale === undefined ? 3 : 3 * scale;
    }

    scaled() {
        return this.get(5);
    }
}

class Deeper extends Impl {
    get(scale?: number) {
        return scale === undefined ? 4 : 4 * scale;
    }
}

class C {
    foo(x: number, y: number, ...z: string[]): number {
        return x + y + z.length;
    }

    callFoo() {
        return this.foo(1, 2, "a", "b");
    }
}

class D extends C {
    foo(): number {
        return 100;
    }

    callOwn() {
        return this.foo();
    }
}

class E extends D {
    foo(x?: number): number {
        return x === undefined ? 200 : 200 + x;
    }
}

function main() {
    const b: Base = new Impl();
    assert(b.twice() == 6, "base slot calls get(scale?) with undefined");

    const i = new Impl();
    assert(i.get() == 3);
    assert(i.get(2) == 6, "own slot keeps the argument");
    assert(i.scaled() == 15, "call in the class body");

    const deep: Base = new Deeper();
    assert(deep.twice() == 8, "base slot calls the latest override");
    const deepImpl: Impl = new Deeper();
    assert(deepImpl.get(2) == 8, "own slot of Impl calls Deeper's");
    assert(deepImpl.scaled() == 20);

    const c: C = new D();
    assert(c.callFoo() == 100, "extra arguments dropped");
    assert(c.foo(1, 2) == 100);
    const d = new D();
    assert(d.foo() == 100);
    assert(d.callOwn() == 100);

    const e: C = new E();
    assert(e.callFoo() == 201, "base slot of E calls foo(x?) with x");
    const ed: D = new E();
    assert(ed.foo() == 200, "D's slot calls E's with undefined");
    assert(new E().foo(5) == 205);

    print("done.");
}

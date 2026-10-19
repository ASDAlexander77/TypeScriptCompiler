// `@final`: a call on a value of a @final class runs the method that class has, directly, and a
// call through its base or its interface still goes through the vtable to the same method.
class B {
    m(): int {
        return 1;
    }

    n(): int {
        return 10;
    }

    get g(): int {
        return 100;
    }
}

interface I {
    m(): int;
}

@final
class F extends B implements I {
    base: int;

    constructor() {
        super();
        this.base = this.n();
    }

    m(): int {
        return 2;
    }

    get g(): int {
        return 200;
    }

    k(): int {
        return this.m() + this.n();
    }
}

@final
class Box<T> {
    constructor(public v: T) {}

    get(): T {
        return this.v;
    }
}

function main() {
    const f = new F();
    assert(f.m() == 2);
    assert(f.n() == 10);
    assert(f.g == 200);
    assert(f.k() == 12);
    assert(f.base == 10);

    assert((<B>f).m() == 2);
    assert((<I>f).m() == 2);
    assert(f instanceof B);

    const fn = f.m;
    assert(fn() == 2);

    let o: F | null = f;
    assert(o?.m() == 2);

    assert(new Box<int>(7).get() == 7);

    print("done.");
}

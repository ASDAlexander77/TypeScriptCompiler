// A call through a non-final base type stays a vtable call, though the object is a @final F.
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
    m(): int {
        return 2;
    }

    get g(): int {
        return 200;
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
    const b: B = new F();
    print(b.m());
}

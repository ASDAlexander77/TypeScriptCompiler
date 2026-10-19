// Only calls on @final types (F, Box<int>), here and in F's own constructor and methods: none of
// them goes through the vtable.
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
    print(f.m(), f.n(), f.g, f.k(), new Box<int>(7).get());
}

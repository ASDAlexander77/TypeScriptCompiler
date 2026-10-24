// A @value interface holding an ordinary interface that returns it, declared before it:
// `Hit.thing: Thing` and `Thing.intersect(): Hit | null`. The interface's method and the class's
// both see Hit with all its fields, so the class implements the interface.
@value
interface Hit {
    readonly thing: Thing;
    readonly d: number;
}

interface Thing {
    intersect(x: number): Hit | null;
    id(): int;
}

class S implements Thing {
    constructor(private n: int) {}

    intersect(x: number): Hit | null {
        if (x > 0) {
            return { thing: this, d: x };
        }

        return null;
    }

    id(): int {
        return this.n;
    }
}

function main() {
    const t: Thing = new S(3);
    const h = t.intersect(2);
    assert(h != null);
    assert(h.thing.id() == 3);
    assert(h.d == 2);
    assert(t.intersect(-1) === null);
    print("done.");
}

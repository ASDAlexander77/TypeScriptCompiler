// `c ? { ... } : null` returned as `T | null` of a @value type: the literal is the T, and the
// ternary is the missing type of T and null, as `if (c) return { ... }; return null;` is.
@value
interface Hit {
    readonly d: number;
}

interface Thing {
    intersect(x: number): Hit | null;
}

class S implements Thing {
    intersect(x: number): Hit | null {
        return x > 0 ? { d: x } : null;
    }
}

function find(x: number): Hit | null {
    return x < 0 ? null : { d: x * 2 };
}

function main() {
    const t: Thing = new S();
    const h = t.intersect(2);
    assert(h != null && h.d == 2);
    assert(t.intersect(-2) === null);

    assert(find(3).d == 6);
    assert(find(-3) === null);

    const k: Hit | null = 1 < 2 ? { d: 5 } : null;
    assert(k != null && k.d == 5);
    print("done.");
}

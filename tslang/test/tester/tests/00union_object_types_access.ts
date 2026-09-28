// A field of a union of object types that share one layout. The union is stored as its base type,
// but the field was read through the union's reference - a 'ts.PropertyRef' the verifier rejected
// (#231).

type Shape = { kind: "circle", r: number } | { kind: "square", r: number };

function kind(s: Shape) {
    if (s.kind === "circle") {
        return 1;
    }

    return 2;
}

function size(s: Shape) {
    return s.r;
}

type Sized = { kind: "circle", r: number | null } | { kind: "square", r: number | null };

function hasSize(s: Sized) {
    if (s.r !== null) {
        return true;
    }

    return false;
}

function main() {
    const c: Shape = { kind: "circle", r: 1 };
    const q: Shape = { kind: "square", r: 2 };
    assert(kind(c) == 1 && kind(q) == 2);
    assert(size(c) == 1 && size(q) == 2);

    const c2: Sized = { kind: "circle", r: 3 };
    const q2: Sized = { kind: "square", r: null };
    assert(hasSize(c2));
    assert(!hasSize(q2));

    print("done.");
}

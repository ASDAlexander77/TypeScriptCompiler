// A field of a union-typed object compared with null or undefined. The discriminated-union
// narrowing took null and undefined for literal constants and crashed the compiler (#231).

interface Circle {
    kind: "circle";
    r: number | null;
}

interface Square {
    kind: "square";
    r: number | null;
}

type Shape = Circle | Square;

function hasR(s: Shape) {
    if (s.r !== null) {
        return true;
    }

    return false;
}

function hasRUndef(s: Shape) {
    if (s.r === undefined) {
        return false;
    }

    return true;
}

function kind(s: Shape) {
    // a real literal must still narrow
    if (s.kind === "circle") {
        return 1;
    }

    return 2;
}

function main() {
    const c: Circle = { kind: "circle", r: 1 };
    const q: Square = { kind: "square", r: null };
    assert(hasR(c));
    assert(!hasR(q));
    assert(hasRUndef(c));
    assert(kind(c) == 1);
    assert(kind(q) == 2);

    print("done.");
}

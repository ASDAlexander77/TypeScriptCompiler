// The library side of @record across a module boundary (own spec 32): `pairOf` and `Maker.pairWith`
// return a borrowing record - a box their caller owns, holding a borrow of each argument `@lends`
// names. The importer sees only these declarations; `@record` tells it the box is its to release.
// See import_own_record.ts.
export class Node {
    constructor(public v: number) {}
}

export interface Pair {
    a: Node;
    b: Node;
}

@lends("a", "b")
@record
export function pairOf(a: Node, b: Node): Pair {
    return { a: a, b: b };
}

@final
export class Maker {
    base = new Node(5);

    @lends("this", "n")
    @record
    pairWith(n: Node): Pair {
        return { a: this.base, b: n };
    }
}

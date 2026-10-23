// The library side of @record across a module boundary (own spec 32): `pairOf` and `Maker.pairWith`
// return a borrowing record - a box their caller owns, holding a borrow of each argument `@lends`
// names. The importer sees only these declarations; `@record` tells it the box is its to release.
// `Maker.weigh` and `Open.look` destroy nothing, and the library says so (`__own_no_drops`); only
// `Maker` is @final. See import_own_record.ts and import_own_record_open_method.ts.
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

    weigh(n: Node) {
        return n.v + this.base.v;
    }
}

export class Open {
    look(n: Node) {
        return n.v;
    }
}

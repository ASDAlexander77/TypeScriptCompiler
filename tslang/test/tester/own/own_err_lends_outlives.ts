// -mm=own rejects: `n` is lent by `firstOf(l)`, a field under `l`; the overwrite of `l.first`
// destroys it before `n.v` reads it. `@lends` is what tells the caller so.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

@lends("l")
export function firstOf(l: List): Node {
    return l.first;
}

function main() {
    const l = new List();
    const n = firstOf(l);
    l.first = new Node(9);
    print(n.v);
}

// -mm=own rejects: a result is either the caller's (`@borrows`) or lent (`@lends`), not both.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

@lends("l")
@borrows("l")
export function firstOf(l: List): Node {
    return l.first;
}

function main() {
    print(firstOf(new List()).v);
}

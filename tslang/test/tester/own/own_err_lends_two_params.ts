// -mm=own rejects: a result is a borrow of one argument.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

@lends("a", "b")
export function pick(a: List, b: List): Node {
    return a.first;
}

function main() {
    print(pick(new List(), new List()).v);
}

// -mm=own rejects: `@lends` names `a`, but the body returns a borrow of `b`.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

@lends("a")
export function pick(a: List, b: List): Node {
    return b.first;
}

function main() {
    print(pick(new List(), new List()).v);
}

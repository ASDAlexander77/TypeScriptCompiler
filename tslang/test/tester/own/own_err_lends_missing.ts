// -mm=own rejects: an exported function returns a borrow of its argument, and a caller in another
// module could not know it; the note says how its signature would.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

export function firstOf(l: List): Node {
    return l.first;
}

function main() {
    print(firstOf(new List()).v);
}

// -mm=own rejects: `@lends` says the result is a borrow of `l`, but the body makes a new one, which
// its callers would then never give back.
export class Node {
    constructor(public v: number) {}
}

export class List {
    first: Node = new Node(1);
}

@lends("l")
export function made(l: List): Node {
    return new Node(l.first.v);
}

function main() {
    print(made(new List()).v);
}

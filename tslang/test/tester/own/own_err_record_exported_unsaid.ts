// An exported function returning a borrowing record has to say `@record`: another module, which sees
// only its declaration, would take the result for a plain borrow and never free the box.
export class Node {
    constructor(public v: number) {}
}

export interface Pair {
    a: Node;
    b: Node;
}

@lends("a", "b")
export function pairOf(a: Node, b: Node): Pair {
    return { a: a, b: b };
}

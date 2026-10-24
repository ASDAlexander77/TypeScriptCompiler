// A library function returning a @value value that borrows its argument and owns nothing: an
// importer cannot be told so yet (own spec 36), and the library says why.
export class Node {
    constructor(public v: number) {}
}

@value
export interface Ref {
    readonly node: Node;
    readonly k: number;
}

@lends("n")
export function wrap(n: Node): Ref {
    return { node: n, k: 1 };
}

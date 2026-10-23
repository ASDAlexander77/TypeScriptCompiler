// `@record` says what `@lends` lends into: alone it names nothing.
class Node {
    constructor(public v: number) {}
}

interface Pair {
    a: Node;
    b: Node;
}

@record
function pairOf(a: Node, b: Node): Pair {
    return { a: a, b: b };
}

function main() {
    const a = new Node(1);
    const b = new Node(2);
    print(pairOf(a, b).a.v);
}

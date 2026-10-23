// `@record` says the caller owns a box, but `first` returns its argument itself: a caller told so
// would free the argument.
class Node {
    constructor(public v: number) {}
}

@lends("a")
@record
function first(a: Node): Node {
    return a;
}

function main() {
    const a = new Node(1);
    print(first(a).v);
}

// The fields of a @value value are readonly: a @value interface's value held in a variable is not
// assigned.
class Node {
    constructor(public n: int) {}
}

@value
interface Ray {
    readonly start: Node;
    readonly dir: Node;
}

function main() {
    const r: Ray = { start: new Node(1), dir: new Node(2) };
    r.start = new Node(3);
    print(r.start.n);
}

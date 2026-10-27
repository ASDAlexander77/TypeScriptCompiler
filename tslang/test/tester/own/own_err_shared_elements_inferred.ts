// -mm=own rejects: type arguments inferred from the constructor's arguments are checked too.
@sharedElements
class Bag<T> {
    constructor(public items: T[]) {}
}

class Node {
    constructor(public n: number) {}
}

function main() {
    const bag = new Bag([new Node(1)]);
    print(bag.items.length);
}

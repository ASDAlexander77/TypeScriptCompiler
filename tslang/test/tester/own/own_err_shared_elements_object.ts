// -mm=own rejects: a @sharedElements container cannot hold an object, whose block has one owner;
// the error names the type argument at the user's `new`, and points at Shared<T>.
@sharedElements
class Bag<T> {
    items: T[] = [];
}

class Node {
    constructor(public n: number) {}
}

function main() {
    const bag = new Bag<Node>();
    print(bag.items.length);
}

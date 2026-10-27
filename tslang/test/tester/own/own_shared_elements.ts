// A @sharedElements container holds values, strings and Shared<T> handles under every memory model;
// under -mm=own an object goes in through Shared<T> (own_err_shared_elements_*).
@sharedElements
class Bag<T> {
    items: T[] = [];

    add(item: T) {
        this.items.push(item);
    }

    get size() {
        return this.items.length;
    }
}

class Node {
    constructor(public n: number) {}
}

function main() {
    const numbers = new Bag<number>();
    numbers.add(1);
    numbers.add(2);

    const names = new Bag<string>();
    names.add("a");

    const node = new Shared(new Node(7));
    const nodes = new Bag<Shared<Node>>();
    nodes.add(node);
    nodes.add(node);

    assert(numbers.size == 2 && names.size == 1 && nodes.size == 2);
    assert(nodes.items[1].value.n == 7);
    print("done.");
}

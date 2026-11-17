// An abstract field of a class type, implemented by a field: under own the accessors over the field
// lend the getter's result and keep the setter's value, as a hand-written pair says with @lends/@keeps.
class Node {
    constructor(public v: number) {}
}

abstract class Holder {
    abstract item: Node | null;
    abstract label: string;
    abstract readonly first: Node;
}

class Box extends Holder {
    item: Node | null = null;
    label = "box";
    first = new Node(7);
}

function fill(h: Holder, n: number) {
    h.item = new Node(n);
    h.label = "filled " + n;
}

function main() {
    const b = new Box();
    const h: Holder = b;
    for (let i = 0; i < 100; i++) fill(h, i);
    const it = h.item;
    if (!it || it.v != 99) throw "item";
    print(h.label);
    print(h.first.v);
    h.item = null;
    if (h.item !== null) throw "cleared";
    print("done.");
}

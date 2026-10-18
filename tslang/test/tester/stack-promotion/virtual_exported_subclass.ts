// A private base with an exported subclass: another module may extend the subclass and override
// visit to keep its argument, so a virtual call through the base cannot be trusted.
class Item {
    v: number;
    constructor(v: number) {
        this.v = v;
    }
}

class Visitor {
    visit(x: Item) {
        return x.v;
    }
}

export class Base extends Visitor {
    visit(x: Item) {
        return x.v + 1;
    }
}

function inner(v: Visitor) {
    const item = new Item(3);
    return v.visit(item);
}

function main() {
    assert(inner(new Base()) == 4);
    assert(inner(new Visitor()) == 3);
    print("done.");
}

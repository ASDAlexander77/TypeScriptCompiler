// -mm=own rejects: an object literal that borrows its field (`{ v: n.items[0] }`) given to a function
// that overwrites that field through its interface (`p.v = ...`). The overwrite would free the
// element, which the literal never owned.
class V {
    constructor(public x: number) {}
}

class N {
    items: V[] = [new V(1)];
}

interface IV {
    v: V;
}

function replace(p: IV) {
    p.v = new V(8);
    return p.v.x;
}

function main() {
    const n = new N();
    print(replace({ v: n.items[0] }), n.items[0].x);
}

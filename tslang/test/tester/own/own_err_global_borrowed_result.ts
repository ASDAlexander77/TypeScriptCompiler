// -mm=own rejects: `child` returns a borrow of its argument's field, so `c` would be a second owner
// of the block `n.c` owns, and `n.c = new V(5)` frees what `c` still reads. A global cannot borrow.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public c: V) {}
}

const n = new N(new V(4));

function child(m: N) {
    return m.c;
}

const c = child(n);

function main() {
    n.c = new V(5);
    print(c.x, n.c.x);
}

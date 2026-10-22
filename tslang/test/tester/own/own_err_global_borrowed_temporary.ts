// -mm=own rejects: `id` returns a borrow of its argument, a temporary the initializer frees at its
// end, so `c` would read a freed block. A global cannot borrow.
class V {
    constructor(public x: number) {}
}

function id(v: V) {
    return v;
}

const c = id(new V(1));

function main() {
    print(c.x);
}

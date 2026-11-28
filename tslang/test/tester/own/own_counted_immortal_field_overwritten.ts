// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// -mm=own rejects: `K.n` is immortal, but only its own block is. Its field `c` is not: assigning
// it destroys the old value, which `c` still reads.
class V {
    constructor(public x: number) {}
}
class N {
    constructor(public c: V) {}
}
class K {
    static n = new N(new V(1));
}
function main() {
    const c = K.n.c;
    K.n.c = new V(2);
    print(c.x);
}

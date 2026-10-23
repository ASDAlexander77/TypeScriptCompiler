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

// -mm=own rejects: `n` is lent to `f` as an interface, and a later argument gives `n` a new value,
// freeing the object before the call reads it (fuzz seed 77055). Given as itself, the same program was
// rejected; seen through the interface view the check missed it, and the call read freed memory.
class V {
    constructor(public x: number) {}
}

class N {
    v: V;
    constructor(x: number) { this.v = new V(x); }
}

interface IX {
    v: V;
}

function mk(i: number) {
    return new N(i);
}

function f(p: IX, q: N) {
    return p.v.x + q.v.x;
}

function main() {
    let n = mk(8);
    print(f(n, (n = mk(3), mk(5))));
}

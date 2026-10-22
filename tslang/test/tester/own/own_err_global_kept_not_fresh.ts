// -mm=own rejects: the constructor keeps `c`, but what it is given is the global `v`, which keeps
// its value, so `n` and `v` would both own one block. Only a value made for the call moves.
class V {
    constructor(public x: number) {}
}

class N {
    constructor(public c: V) {}
}

const v = new V(1);
const n = new N(v);

function main() {
    print(n.c.x, v.x);
}

// -mm=own rejects: an object literal that borrows an element of `n` (`{ v: n.items[0] }`) given to a
// call whose later argument gives `n` a new value, which frees the element the literal borrows. Inside
// a `try` the calls are invokes, and the literal's use by the call was not seen.
class V {
    constructor(public x: number) {}
}

class N {
    items: V[];
    constructor(x: number) {
        this.items = [new V(x)];
    }
}

interface IV {
    v: V;
}

function mk(x: number) {
    return new N(x);
}

function read(p: IV, q: N) {
    return p.v.x + q.items[0].x;
}

function main() {
    let n = mk(7);
    let s = 0;
    try {
        s += read({ v: n.items[0] }, (n = mk(1), n));
    } finally {
    }

    print(s);
}

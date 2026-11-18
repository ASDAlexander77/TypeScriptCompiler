// -mm=own: a field read through a `let` given another local's object, stored into an owning field
// while the first local is still read after: the store would take the value out of the object both
// name, and the first one would read the emptied field (fuzz 956382 crashed)
class V {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class N {
    v: V;
    constructor(v: V) {
        this.v = v;
    }
}

@noinline
function mk(i: number): N {
    return new N(new V(i));
}

function main() {
    const n = mk(3);
    let copy: N = n;
    const o: N = new N(new V(7));
    o.v = copy.v;
    print(n.v.x);
}

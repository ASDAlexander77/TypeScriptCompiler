// -mm=own: a field taken through a `let` given another local's object, where that local is not read
// after the take: the value moves out of the field (fuzz 956382 refuses only the case read after)
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
    let sum: number = 0;
    for (let i = 0; i < 300; i++) {
        const n = mk(i);
        sum += n.v.x;
        let copy: N = n;
        const o: N = new N(new V(7));
        o.v = copy.v;
        sum += o.v.x;
    }
    assert(sum == 300 * 299, "sum");
    print("done.");
}

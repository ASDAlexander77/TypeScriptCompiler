// -mm=own: a local moved into a call on one path of a loop body, released where the ways join, and
// released on a `continue` of its own (fuzz 961334): the join's release went onto every edge into
// the rest of the loop body, the `continue` too - released twice there, the heap broke.
class V {
    x: number;
    a: number[];
    constructor(x: number) {
        this.x = x;
        this.a = [x];
    }
}

class N {
    v: V;
    items: V[];
    constructor(v: V) {
        this.items = [new V(v.x)];
        this.v = v;
    }
}

@noinline
function mk(i: number): N {
    return new N(new V(i));
}

@noinline
function take(p: N): N {
    p = mk(3);
    return p;
}

function main() {
    let sum: number = 0;
    for (let i = 0; i < 300; i++) {
        const n = mk(9);
        if (i % 3 == 0) {
            if (i % 2 == 1) {
                continue;
            }

            sum += take(n).v.x;
        }
    }

    assert(sum == 3 * 50, "sum");
    print("done.");
}

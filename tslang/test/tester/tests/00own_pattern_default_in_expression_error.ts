// -mm=own: a destructuring assignment with a default that makes a new value, inside an expression (`c ? ({ p =
// mk(9) } = r) : ...`), is refused as before, with the ownership error - it failed with an internal one ("operand #0
// does not dominate this use"): the default's slot, made inside the conditional, is not the scope's.
class N {
    x: number;
    constructor(x: number) {
        this.x = x;
    }
}

class R {
    p?: N;
}

function mk(i: number): N {
    return new N(i);
}

function main() {
    const r = new R();
    let p: N = mk(0);
    for (let i = 0; i < 4; i++) {
        const v = i % 3 == 0 ? (({ p = mk(9) } = r), p.x) : 0;
        print(v);
    }

    print("done.");
}

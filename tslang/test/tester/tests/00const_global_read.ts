// A `const` read out of a global keeps what it read alive when the global is given a new value,
// here or in a call. Folded, the const was the read itself and held no reference of its own: under
// rc the overwrite freed the block it still read, and a value read through it and lent to a call
// that assigns the global (`run(n.v)`) was retained and released after the block was gone.
// Allocations of the same size reuse anything freed before each read.
class V {
    tag = "";
    constructor(public x: number) {
        this.tag = "v" + x;
    }

    get() {
        return this.x;
    }
}

class N {
    v: V;
    constructor(x: number) {
        this.v = new V(x);
    }
}

interface IV {
    x: number;
    get(): number;
}

class Junk {
    a = -1;
    b = -2;
    tag = "junk";
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

let G = new N(1);
let GV = new V(1);

function assignG(x: number) {
    G = new N(x);
    churn();
}

// assigns the global its argument was read out of
function run(v: V, x: number) {
    assignG(100 + x);
    return v.x == x && v.tag == "v" + x;
}

function reassignedHere(i: number) {
    G = new N(i);
    const n = G;
    G = new N(9);
    churn();
    assert(n.v.x == i && n.v.tag == "v" + i, "a const read of a global, reassigned here");
}

function reassignedInCall(i: number) {
    G = new N(i);
    const n = G;
    assignG(9);
    assert(n.v.x == i && n.v.get() == i, "a const read of a global, reassigned in a call");
}

function lentFromConst(i: number) {
    G = new N(i);
    const n = G;
    for (let j = 0; j < 3; j++) {
        assert(run(n.v, i), "a field of a const read of a global, lent to a call that assigns it");
        try {
            churn();
        } finally {
            assert(n.v.tag == "v" + i, "read in a finally");
        }
    }
}

function views(i: number) {
    GV = new V(i);
    const iv: IV = GV;
    const m = GV.get;
    GV = new V(9);
    churn();
    assert(iv.x == i && iv.get() == i && m() == i, "an interface and a bound method over a global");
}

function main() {
    for (let i = 2; i < 5; i++) {
        reassignedHere(i);
        reassignedInCall(i);
        lentFromConst(i);
        views(i);
    }

    churn();
    assert(G.v.x > 100 && GV.x == 9, "the globals' last values");
    print("done.");
}

// A `const` read out of a field keeps what it read alive when the field is given a new value.
// Folded, the const was the read itself and held no reference of its own: under rc the overwrite
// freed the block it still read. Allocations of the same size reuse anything freed before each
// read.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(1);
    n: C | null = new C(2);
}

class Junk {
    a = -1;
    b = -2;
}

function churn() {
    let keep: Junk[] = [];
    for (let i = 0; i < 1000; i++) keep.push(new Junk());
    return keep.length;
}

function field(i: number) {
    const h = new H();
    h.c = new C(i);
    const c = h.c;
    h.c = new C(9);
    churn();
    return c.x == i && h.c.x == 9;
}

function nullableField(i: number) {
    const h = new H();
    h.n = new C(i);
    const c = h.n;
    h.n = null;
    churn();
    return c != null && c.x == i;
}

function readInFinally(h: H) {
    const c = h.c;
    let x = 0;
    try {
        h.c = new C(9);
        return x;
    } finally {
        churn();
        x = c.x;
        assert(x == 1, "the read survives the field's overwrite through finally");
    }
}

function main() {
    for (let i = 0; i < 20; i++) {
        assert(field(i), "a field read");
        assert(nullableField(i), "a nullable field read");
    }

    readInFinally(new H());
    print("done.");
}

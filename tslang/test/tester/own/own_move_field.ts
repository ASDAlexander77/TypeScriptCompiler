// -mm=own, phase 1: a `let` moved into a field. The field owns it; the holder's release routine
// destroys it; the `let`'s own release goes.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        keep.push(c);
    }
}

function store(i: number) {
    const h = new H();
    let a = new C(i);
    a.v.push(i);
    h.c = a;
    churn();
    const c = h.c;
    return c ? c.x + c.v.length : -1000;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += store(i % 10);
    }

    assert(t == 550000);
    print("done.");
}

// -mm=own, phase 1: a `let` returned. At the affine level the return is a store into the return
// slot, and the `let`'s scope-exit release sits between rc's retain and that store - after the
// move, so it goes.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        keep.push(c);
    }
}

function make(i: number) {
    let a = new C(i);
    a.v.push(i);
    a.x = a.x + 1;
    return a;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const c = make(i % 10);
        churn();
        t += c.x + c.v.length;
    }

    assert(t == 650000);
    print("done.");
}

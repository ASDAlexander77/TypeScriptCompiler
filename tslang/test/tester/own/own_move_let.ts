// -mm=own, phase 1: a `let` hands its value to another `let` and is not used again. The first
// slot's release goes; the second slot's release destroys the value once.
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

function relay(i: number) {
    let a = new C(i);
    a.v.push(i);
    let b = a;
    churn();
    return b.x + b.v.length;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += relay(i % 10);
    }

    assert(t == 550000);
    print("done.");
}

// -mm=own, phase 2: two `let`s of one `let`, both borrowing it; neither may move it.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

function twice(i: number) {
    let a = new C(i);
    a.v.push(i);
    let b = a;
    let c = a;
    churn();
    return b.x + c.v.length + a.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += twice(i % 10);
    }

    assert(t == 1000000);
    print("done.");
}

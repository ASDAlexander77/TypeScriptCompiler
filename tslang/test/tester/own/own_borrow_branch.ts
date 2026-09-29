// -mm=own, phase 2: a `let` declared from a value on one branch only borrows it; phase 1 had to
// reject it as "moved on some paths only".
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

function pick(i: number) {
    const a = new C(i);
    a.v.push(i);
    let t: number = 0;
    if (i % 2 == 0) {
        let b = a;
        t = b.x + b.v.length;
    }

    churn();
    return t + a.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += pick(i % 10);
    }

    assert(t == 700000);
    print("done.");
}

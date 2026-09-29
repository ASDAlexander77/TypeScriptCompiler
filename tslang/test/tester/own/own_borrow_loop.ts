// -mm=own, phase 2: a `let` inside a loop borrows a value made before the loop (spec 2.6), and
// a `let` that borrows an owner reassigned later in the same body borrows the new value on the
// next iteration.
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

function fromOutside(n: number) {
    const a = new C(n);
    a.v.push(n);
    let t: number = 0;
    for (let i = 0; i < 3; i++) {
        let b = a;
        t += b.x + b.v.length;
    }

    churn();
    return t + a.x;
}

function reassigned(n: number) {
    let a = new C(n);
    let t: number = 0;
    for (let i = 0; i < 3; i++) {
        let b = a;
        t += b.x;
        a = new C(b.x + 1);
    }

    churn();
    return t + a.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 10000; i++) {
        t += fromOutside(i % 10) + reassigned(i % 10);
    }

    assert(t == 450000);
    print("done.");
}

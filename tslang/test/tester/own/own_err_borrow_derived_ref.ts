// -mm=own, phase 2 rejects: the loop iterates `b.v`, a field of what `b` borrows, through one
// reference into it that every iteration reads - after the body has overwritten `a`.
class C {
    v: number[] = [];
    constructor(public x: number) {}
    m() { return this.x; }
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        junk.push(c);
    }
}

function main() {
    let a = new C(1);
    a.v.push(1);
    a.v.push(2);
    a.v.push(3);
    let b = a;
    let t: number = 0;
    for (const x of b.v) {
        a = new C(2);
        churn();
        t += x;
    }

    print(t);
}

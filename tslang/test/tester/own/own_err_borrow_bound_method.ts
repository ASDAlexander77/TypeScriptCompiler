// -mm=own, phase 2 rejects: `f` is `b.m` bound to what `b` borrows, so calling it after `a` is
// overwritten reads the old `a`. What a borrower's reads produce is the borrower too.
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
    let b = a;
    const f = b.m;
    a = new C(2);
    churn();
    print(f());
}

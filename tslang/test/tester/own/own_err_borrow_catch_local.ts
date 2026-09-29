// -mm=own, phase 2 rejects: a catch clause's locals own nothing, so `x` is one more name for what
// `b` borrows - and it is read after `a` is overwritten.
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
    let b = a;
    try {
        throw 1;
    } catch (e) {
        let x = b;
        a = new C(2);
        churn();
        print(x.x, x.v.length);
    }
}

// -mm=own, phase 2 rejects: `f` is `b.m`, a method value of `b`, which only borrows `a` (`a` is given
// to two `let`s, so neither takes its value over: each borrows it). A method value is its object, so
// `f` would hold what `b` borrows past the borrow; it is refused where it is made. (Before the method
// value was its object: refused where `f()` ran after `a` was overwritten.)
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
    let c = a;
    print(a.x, c.x);
    const f = b.m;
    a = new C(2);
    churn();
    print(f());
}

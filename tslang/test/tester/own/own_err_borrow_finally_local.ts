// -mm=own, phase 2 rejects: the same through a finally clause's local. (`a` is given to two
// `let`s, so neither takes its value over: each borrows it. One `let` would take it over, spec 29.)
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
    let c = a;
    print(a.x, c.x);
    try {
        print(b.x);
    } finally {
        let x = b;
        a = new C(2);
        churn();
        print(x.x, x.v.length);
    }
}

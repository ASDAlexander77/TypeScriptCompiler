// -mm=own, phase 3 rejects: a value read out of a field is borrowed; storing it into another
// field would give it a second owner.
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

class H {
    c: C = new C(0);
    n: number = 0;
    s: string = "a";
}

function main() {
    const h = new H();
    const h2 = new H();
    h2.c = h.c;
    print(h2.c.x);
}

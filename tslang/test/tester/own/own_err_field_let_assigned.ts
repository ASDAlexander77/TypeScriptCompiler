// -mm=own, phase 3 rejects: `c` borrows `h.c`, so it cannot be assigned.
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
    let c = h.c;
    print(c.x);
    c = new C(5);
    print(c.x);
}

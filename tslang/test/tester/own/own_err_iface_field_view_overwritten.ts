// -mm=own rejects: `c` borrows the class field `h.c`, which is overwritten through an interface
// view of `h` before `c` is read (spec 28).
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

interface HI { c: C; }
class H { c: C = new C(0); }

function main() {
    const h = new H();
    const hi: HI = h;
    const c = h.c;
    hi.c = new C(9);
    churn();
    print(c.x);
}

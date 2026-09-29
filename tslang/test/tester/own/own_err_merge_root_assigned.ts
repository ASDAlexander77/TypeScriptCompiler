// -mm=own, phase 3 rejects: `r` is `a` or `b`, so `r.c` borrows from both; assigning `a` destroys
// the old `a` and its `c`.
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
}

function run(i: number) {
    let a = new H();
    a.c = new C(5);
    let b = new H();
    b.c = new C(6);
    let z = new H();
    z.c = new C(7);
    const r = i > 0 ? a : b;
    const c = r.c;
    a = z;
    const junk: number[][] = [];
    for (let k = 0; k < 64; k++) {
        junk.push([999, 999, 999, 999]);
    }

    return c.x + junk.length;
}

function main() {
    print(run(1));
}

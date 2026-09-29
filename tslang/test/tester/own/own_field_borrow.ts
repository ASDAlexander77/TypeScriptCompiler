// -mm=own, phase 3: `const c = h.c` borrows the field. Stores into other fields - a number, a
// string - cannot destroy it, and neither can a call that is not given `h`; the overwrite of `h.c`
// comes after its last use.
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
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const h = new H();
        h.c = new C(i % 10);
        h.c.v.push(i % 10);
        const c = h.c;
        h.n = h.n + c.x;
        h.s = "b" + i;
        churn();
        t += c.x + c.v.length + h.n;
        h.c = new C(1);
        t += h.c.x;
    }

    assert(t == 1100000);
    print("done.");
}

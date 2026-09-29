// -mm=own, phase 3: each iteration borrows `h.c` afresh and overwrites it after the last use; the
// path back to the next iteration's use passes through the next read.
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
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        const c = h.c;
        churn();
        t += c.x;
        h.c = new C(i % 10);
    }

    assert(t == 449991);
    print("done.");
}

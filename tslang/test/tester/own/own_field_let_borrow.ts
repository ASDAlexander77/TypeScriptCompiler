// -mm=own, phase 3: `let c = h.c` borrows the field - no reference taken, none given back - and
// its uses end before the overwrite.
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
        let c = h.c;
        churn();
        t += c.x + c.v.length;
        h.c = new C(1);
        t += h.c.x;
    }

    assert(t == 650000);
    print("done.");
}

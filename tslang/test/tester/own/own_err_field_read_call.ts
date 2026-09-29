// -mm=own, phase 3 rejects: `c` borrows `h.c`, and a call given `h` may overwrite it.
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

function reset(h: H) {
    h.c = new C(9);
}

function main() {
    const h = new H();
    const c = h.c;
    reset(h);
    churn();
    print(c.x);
}

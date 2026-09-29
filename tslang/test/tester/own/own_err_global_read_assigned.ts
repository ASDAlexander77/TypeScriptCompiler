// -mm=own, phase 3 rejects: `c` borrows a field of the global `g`, and assigning `g` destroys the
// old value with it.
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

let g = new H();

function main() {
    const c = g.c;
    g = new H();
    churn();
    print(c.x);
}

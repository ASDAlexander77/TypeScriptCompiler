// -mm=own, phase 3 rejects: `c` is `n.c` or `m.c` through a merge, and `n.c` is overwritten before
// the read.
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

class N {
    c: C | null = null;
}

function main() {
    const n = new N();
    const m = new N();
    n.c = new C(1);
    const c = n.c ?? m.c;
    n.c = new C(2);
    churn();
    if (c) {
        print(c.x);
    }
}

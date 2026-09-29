// -mm=own, phase 3: a borrow passed on through a merge (`??`) or stored into a local that owns
// nothing (`for (x of a)` into an outer `let`) is still that borrow, bounded by its container.
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


class N {
    c: C | null = null;
}

function one(n: N, m: N) {
    const c = n.c ?? m.c;
    let t: number = 0;
    if (c) {
        t += c.x;
    }

    return t;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 10000; i++) {
        const n = new N();
        const m = new N();
        m.c = new C(i % 10);
        t += one(n, m);
        const arr: C[] = [new C(1), new C(2)];
        let x: C | number;
        for (x of arr) {
            churn();
            if (typeof x !== "number") {
                t += x.x;
            }
        }
    }

    assert(t == 75000);
    print("done.");
}

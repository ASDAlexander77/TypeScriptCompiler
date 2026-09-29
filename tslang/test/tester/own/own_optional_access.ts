// -mm=own, phase 3: testing an optional (`d?.x`, `if (d)`) reads it; it does not take it.
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


function one(i: number) {
    const c = new C(i);
    const d: C | undefined = c;
    churn();
    let t: number = 0;
    if (d) {
        t += d.x;
    }

    d?.x;
    d?.v;
    if (d !== undefined) {
        t += d.x + d.v.length;
    }

    return t;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += one(i % 10);
    }

    assert(t == 900000);
    print("done.");
}

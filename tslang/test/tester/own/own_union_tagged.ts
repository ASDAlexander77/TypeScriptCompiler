// -mm=own, phase 3: a tagged union owns its class payload. Making the union, reading the payload
// back after `typeof` and overwriting it with a string literal are views of that payload.
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

function one(i: number) {
    let t: number = 0;
    let u: C | string = new C(i);
    if (typeof u !== "string") {
        t += u.x;
    }

    const c = new C(i + 1);
    let w: C | string = c;
    churn();
    if (typeof w !== "string") {
        t += w.x;
    }

    u = "s";
    if (typeof u === "string") {
        t += 1;
    }

    return t;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += one(i % 10);
    }

    assert(t == 1100000);
    print("done.");
}

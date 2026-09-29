// -mm=own, phase 3: an optional local owns its class value; testing it and narrowing it with `!`
// read it through the view.
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
    let u: C | undefined = new C(i);
    churn();
    let t: number = 0;
    if (u !== undefined) {
        t += u.x;
    }

    t += u!.x;
    u = undefined;
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

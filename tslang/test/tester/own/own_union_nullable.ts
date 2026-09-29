// -mm=own, phase 3: a nullable local owns its class value. Widening it to `C | null`, testing it
// and narrowing it back with `!` are views of the one block, not second owners.
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
    let u: C | null = new C(i);
    u!.v.push(i);
    churn();
    let t: number = 0;
    if (u) {
        t += u.x + u.v.length;
    }

    t += u!.x;
    u = null;
    return t;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += one(i % 10);
    }

    assert(t == 1000000);
    print("done.");
}

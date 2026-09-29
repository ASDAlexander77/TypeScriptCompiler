// -mm=own, phase 1: a value moved in a branch that returns. The move reaches only that branch's
// own release, which goes, and the release on the path that falls through stays.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        keep.push(c);
    }
}

function pick(i: number) {
    const a = new C(i);
    a.v.push(i);
    if (i % 2 == 0) {
        let b = a;
        churn();
        return b.x + b.v.length;
    }

    churn();
    return a.x;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += pick(i % 10);
    }

    assert(t == 500000);
    print("done.");
}

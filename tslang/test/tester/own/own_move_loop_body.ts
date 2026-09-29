// -mm=own, phase 1: a move inside a loop body whose source is declared in that body. Each
// iteration makes a new `a`, so the move is not "made outside the loop" (spec 2.6). The values
// outlive the loop in `keep`, and are read after churn(), so a release of `a` left behind would
// free a block `keep` still holds and the sum would come out wrong.
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

function main() {
    const keep: C[] = [];
    for (let i = 0; i < 1000; i++) {
        let a = new C(i % 10);
        a.v.push(i);
        let b = a;
        keep.push(b);
    }

    churn();
    let t: number = 0;
    for (const c of keep) {
        t += c.x + c.v.length;
    }

    assert(t == 5500);
    print("done.");
}

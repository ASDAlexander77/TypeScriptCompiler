// -mm=own, phase 2: a nullable `let` borrowing a class value, tested for null. The test reads a
// boolean out of the borrower, which keeps nothing.
class C {
    v: number[] = [];
    constructor(public x: number) {}
    m() { return this.x; }
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
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        let a = new C(i % 10);
        a.v.push(3);
        {
            let b: C | null = a;
            if (b) {
                t += b.x + b.v.length;
            }
        }

        churn();
        t += a.x;
    }

    assert(t == 1000000);
    print("done.");
}

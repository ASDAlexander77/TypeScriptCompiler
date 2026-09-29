// -mm=own, phase 2: a `let` in an inner scope borrows a value that is used again after that
// scope. The `let` takes no reference and releases nothing; the owner's release destroys it once.
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

function fromConst(i: number) {
    const a = new C(i);
    a.v.push(i);
    {
        let b = a;
        b.x = b.x + 1;
    }

    churn();
    return a.x + a.v.length;
}

function fromLet(i: number) {
    let a = new C(i);
    a.v.push(i);
    {
        let b = a;
        b.v.push(i);
    }

    churn();
    return a.x + a.v.length;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        t += fromConst(i % 10) + fromLet(i % 10);
    }

    assert(t == 1300000);
    print("done.");
}

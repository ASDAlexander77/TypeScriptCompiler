// -mm=own, phase 3 rejects: `e` borrows `arr[0]`, and the `pop` - whose result is given back at
// the end of its block - destroys it.
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
    const arr: C[] = [];
    arr.push(new C(1));
    const e = arr[0];
    {
        arr.pop();
    }

    churn();
    print(e.x);
}

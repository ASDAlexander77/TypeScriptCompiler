// -mm=own, phase 3 rejects: `e` borrows `arr[0]`, and the element store destroys it.
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
    arr[0] = new C(2);
    churn();
    print(e.x);
}

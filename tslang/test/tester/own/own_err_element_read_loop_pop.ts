// -mm=own, phase 3 rejects: the loop's `f` borrows an element that the body pops.
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
    for (const f of arr) {
        {
            arr.pop();
        }

        churn();
        print(f.x);
    }
}

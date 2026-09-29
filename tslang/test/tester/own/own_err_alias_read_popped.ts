// -mm=own, phase 3 rejects: `x` owns nothing, so the loop stores a borrow of each element into it,
// and the `pop` destroys the element `x` still names.
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
    const arr: C[] = [new C(1)];
    let x: C | number;
    for (x of arr) {
    }

    {
        arr.pop();
    }

    churn();
    if (typeof x !== "number") {
        print(x.x);
    }
}

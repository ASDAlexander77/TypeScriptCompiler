// -mm=own, phase 3: `let e = arr[j]` in a loop borrows each element in turn.
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
    let t: number = 0;
    for (let i = 0; i < 1000; i++) {
        const arr: C[] = [];
        for (let j = 0; j < 10; j++) {
            arr.push(new C(j));
        }

        for (let j = 0; j < 10; j++) {
            let e = arr[j];
            churn();
            t += e.x;
        }

        arr.pop();
    }

    assert(t == 45000);
    print("done.");
}

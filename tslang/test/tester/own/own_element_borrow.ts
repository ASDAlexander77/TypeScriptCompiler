// -mm=own, phase 3: `const e = arr[3]` and `for (const f of arr)` borrow elements. Pushing onto
// the same array or another one cannot destroy an element, and neither can a call that is not
// given `arr`; the `pop` comes after the last use.
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

        const e = arr[3];
        const out: C[] = [];
        out.push(new C(e.x));
        arr.push(new C(10));
        churn();
        t += e.x + out.length + arr.length;
        for (const f of arr) {
            t += f.x;
        }

        arr.pop();
        t += arr.length;
    }

    assert(t == 80000);
    print("done.");
}

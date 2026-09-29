// -mm=own, phase 4: an `any` box owns what it holds. Boxing moves the value in and makes a box
// with one owner - a `let`, a folded `const`, or the temporary passed to a call - and unboxing
// (`___unbox`, whose result borrows its argument) borrows the payload for as long as the box
// lives. Every value is read after a `churn()`.
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

function kind(v: any) {
    return typeof v == "class" ? 1 : 0;
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 100000; i++) {
        let a: any = new C(i % 10);
        const b: any = new C(1);
        const ca = <C>a;
        churn();
        t += ca.x + (<C>b).x + kind(new C(2)) + kind(a);
    }

    assert(t == 750000);
    print("done.");
}

// An array literal given straight to a call is a new value, as `new C()` is: under own it is given
// back after the call when nothing takes it, moved into a callee that keeps it, and kept alive while
// a record the callee returns borrows it. Under own it was never given back at all - 400 bytes a
// call for `len([...])`, hidden by the optimizer only where the array went nowhere else. Each value
// is read after churn() has reused anything freed too early.
interface A {
    a: number[];
}

function churn() {
    const keep: number[][] = [];
    for (let i = 0; i < 20; i++) keep.push([77777, 77777, 77777]);
    return keep.length;
}

function len(a: number[]) {
    return a.length;
}

function wrap(a: number[]): A {
    return { a };
}

function grow(a: number[]): A {
    const r: A = { a };
    for (let j = 0; j < 10; j++) a.push(j);
    return r;
}

const kept: number[][] = [];

function keep(a: number[]) {
    kept.push(a);
}

function main() {
    let s: number = 0;
    for (let i = 0; i < 1000; i++) {
        s += len([i, 1, 2]);

        const w = wrap([i, 4, 5, 6]);
        churn();
        s += w.a.length + (w.a[0] == i ? 1 : 0);

        const g = grow([i]);
        churn();
        s += g.a.length + g.a[10];

        if (i % 100 == 0) keep([i, i]);
    }

    churn();
    assert(kept.length == 10 && kept[9][1] == 900, "the arrays a callee keeps");
    assert(s == 1000 * (3 + 5 + 11 + 9), "every array read after churn");
    print("done.");
}

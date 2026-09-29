// -mm=own, phase 1: moves on the control-flow shapes phase 1 opened, each read after churn():
// a loop body that moves out of its own `let` and leaves early by `break` or `continue`, a
// return from inside a loop of a value made in the body, and one made before the loop.
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

function collect() {
    const keep: C[] = [];
    for (let n = 0; n < 100; n++) {
        let a = new C(n % 10);
        a.v.push(n);
        if (n == 70) break;
        if (n % 3 == 0) continue;
        let b = a;
        keep.push(b);
    }

    churn();
    let t: number = 0;
    for (const c of keep) {
        t += c.x + c.v.length;
    }

    return t;
}

// made in the loop body, returned from it
function findInBody(n: number) {
    for (let i = 0; i < 10; i++) {
        let a = new C(i);
        a.v.push(i);
        if (i == n) return a;
    }

    return new C(-1);
}

// made before the loop, returned from inside it: the return is the move, reached once
function findBefore(n: number) {
    let a = new C(n);
    a.v.push(n);
    for (let i = 0; i < 10; i++) {
        if (i == n) return a;
    }

    return new C(-1);
}

function main() {
    let t: number = 0;
    for (let k = 0; k < 1000; k++) {
        const c = findInBody(k % 10);
        const d = findBefore(k % 12);
        churn();
        t += c.x + c.v.length + d.x + d.v.length;
    }

    assert(collect() == 253);
    assert(t == 9909);
    print("done.");
}

// -mm=own (#528): a `let` whose value moved is assigned a new value and owns that one. The
// assignment gives nothing up - what the slot held is its receiver's now - and the slot's release
// at scope exit destroys the new value once.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function churn() {
    const keep: C[] = [];
    for (let i = 0; i < 64; i++) {
        const c = new C(999);
        c.v.push(99);
        keep.push(c);
    }
}

function x(c: C | null) {
    return c ? c.x + c.v.length : -1000;
}

// into another `let`, then a new value
function relay(i: number) {
    let a = new C(i);
    a.v.push(i);
    let b = a;
    a = new C(i + 1);
    a.v.push(i);
    churn();
    return b.x + b.v.length + a.x + a.v.length;
}

// into a field, then a new value
function store(i: number) {
    const h = new H();
    let a = new C(i);
    a.v.push(i);
    h.c = a;
    a = new C(i + 1);
    a.v.push(i);
    churn();
    return x(h.c) + a.x + a.v.length;
}

// two values in turn, each moved into its own field, then a third the slot keeps
function twice(i: number) {
    const h = new H();
    const g = new H();
    let a = new C(i);
    a.v.push(i);
    h.c = a;
    a = new C(i + 1);
    a.v.push(i);
    g.c = a;
    a = new C(i + 2);
    a.v.push(i);
    churn();
    return x(h.c) + x(g.c) + a.x + a.v.length;
}

// moved and given a new value on every iteration of a loop: each one moves into the array
function loop(i: number) {
    const keep: C[] = [];
    let a = new C(i);
    for (let k = 0; k < 3; k++) {
        a.v.push(k);
        keep.push(a);
        a = new C(i + k + 1);
    }

    churn();
    return keep[0].x + keep[2].x + keep[2].v.length + a.x;
}

function main() {
    let t1: number = 0;
    let t2: number = 0;
    let t3: number = 0;
    let t4: number = 0;
    for (let i = 0; i < 20000; i++) {
        t1 += relay(i % 10);
        t2 += store(i % 10);
        t3 += twice(i % 10);
        t4 += loop(i % 10);
    }

    // relay and store: (i + 1) + (i + 2), summed over i = 0..9 two thousand times
    assert(t1 == 2000 * (2 * 45 + 30));
    assert(t2 == 2000 * (2 * 45 + 30));
    // twice: (i + 1) + (i + 2) + (i + 3)
    assert(t3 == 2000 * (3 * 45 + 60));
    // loop: i + (i + 2) + 1 + (i + 3)
    assert(t4 == 2000 * (3 * 45 + 60));
    print("done.");
}

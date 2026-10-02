// -mm=own, phase 7: a record built to be returned - here the shape a generator's `next` returns -
// takes the fresh values stored into its fields, and the caller owns the record. Each case reads
// through a churn() after the call.
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

function make(k: number) {
    const c = new C(k);
    c.v.push(k);
    return { value: c, done: false };
}

function fresh(k: number) {
    return { value: new C(k), done: k > 5 };
}

function main() {
    let t = 0;
    for (let i = 0; i < 20000; i++) {
        const n = i % 10;
        let r = make(n);
        churn();
        t += r.value.x + r.value.v.length;

        // assigned again: the first record is released
        r = fresh(n);
        churn();
        t += r.value.x + (r.done ? 1 : 0);
    }

    assert(t == 208000, "t");
    print("done.");
}

// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own rejects: `y` borrows `h.c`, and `reset`, called inside a try body (a `ts.Invoke`),
// overwrites it. A call there was not a call to the passes, so this compiled and read freed memory.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(1);
}

function churn() {
    const junk: C[] = [];
    for (let i = 0; i < 64; i++) {
        junk.push(new C(999));
    }
}

function reset(h: H) {
    h.c = new C(9);
}

function read(h: H) {
    const y = h.c;
    try {
        reset(h);
    } catch (e) {
    }

    churn();
    print(y.x);
}

function main() {
    read(new H());
}

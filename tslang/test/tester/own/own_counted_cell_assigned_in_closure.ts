// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own, phase 5 rejects: `y` borrows a field of the value in `a`'s cell, and the closure
// assigns `a`.
class C {
    constructor(public x: number) {}
}

class H {
    constructor(public c: C) {}
}

function main() {
    let a = new H(new C(1));
    const f = () => {
        a = new H(new C(2));
    };
    const y = a.c;
    f();
    print(y.x);
}

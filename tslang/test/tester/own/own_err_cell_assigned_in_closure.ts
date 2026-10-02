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

// -mm=own, phase 5 rejects: `y` borrows a field of the value in `a`'s cell, and assigning `a`
// destroys that value.
class C {
    constructor(public x: number) {}
}

class H {
    constructor(public c: C) {}
}

function main() {
    let a = new H(new C(1));
    const f = () => a.c.x;
    const y = a.c;
    a = new H(new C(2));
    print(y.x, f());
}

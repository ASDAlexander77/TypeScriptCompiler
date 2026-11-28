// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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

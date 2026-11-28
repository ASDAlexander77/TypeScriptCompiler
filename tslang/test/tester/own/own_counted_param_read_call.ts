// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own, phase 3 rejects: `c` borrows a field of a parameter, which this function does not own;
// any call may reach its owner - here through a global - and overwrite the field.
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

class H {
    c: C = new C(0);
    n: number = 0;
    s: string = "a";
}

let g = new H();

function reset() {
    g.c = new C(9);
}

function read(h: H) {
    const c = h.c;
    reset();
    churn();
    print(c.x);
}

function main() {
    read(g);
}

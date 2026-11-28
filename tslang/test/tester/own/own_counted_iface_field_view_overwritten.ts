// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// -mm=own rejects: `c` borrows the class field `h.c`, which is overwritten through an interface
// view of `h` before `c` is read (spec 28).
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

interface HI { c: C; }
class H { c: C = new C(0); }

function main() {
    const h = new H();
    const hi: HI = h;
    const c = h.c;
    hi.c = new C(9);
    churn();
    print(c.x);
}

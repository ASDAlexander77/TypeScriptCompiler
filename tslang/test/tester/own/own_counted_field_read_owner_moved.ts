// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
// -mm=own, phase 3 rejects: `c` borrows `h.c`; `h` then moves into `keep`, whose `pop` destroys
// it - a borrow ends where its root moves.
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

function main() {
    const keep: H[] = [];
    const h = new H();
    const c = h.c;
    keep.push(h);
    {
        keep.pop();
    }

    churn();
    print(c.x);
}

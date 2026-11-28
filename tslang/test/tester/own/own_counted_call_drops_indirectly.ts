// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own, phase 4 rejects: `reset` destroys nothing itself, but it calls `clear`, which assigns
// the global that owns what `c` borrows. The fact goes up the call graph.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);
}

let g = new H();

function clear() {
    g.c = new C(9);
}

function reset() {
    clear();
}

function read(h: H) {
    const c = h.c;
    reset();
    print(c.x);
}

function main() {
    read(g);
}

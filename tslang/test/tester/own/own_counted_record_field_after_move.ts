// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// A field read out of a record is a read of the record: given away to a callee that keeps it, the record
// is gone, and the field read after that reads freed memory.
class N {
    constructor(public x: number) {}
}

function mk(i: number) {
    return new N(i);
}

class K {
    recs: { a: N; b: N }[] = [];
}

function keep(k: K, r: { a: N; b: N }) {
    k.recs.push(r);
}

function main() {
    const k = new K();
    const r = { a: mk(1), b: mk(2) };
    const a = r.a;
    keep(k, r);
    print(a.x);
}

// A field of a record the function made is the record's: returning it alone would outlive the record.
class N {
    constructor(public x: number) {}
}

function mk(i: number) {
    return new N(i);
}

function first() {
    const r = { a: mk(1), b: mk(2) };
    return r.a;
}

function main() {
    print(first().x);
}


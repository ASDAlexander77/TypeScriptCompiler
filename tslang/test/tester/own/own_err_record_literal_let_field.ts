// -mm=own rejects (spec 36): `r` holds the value of the `let` `v`, which `v` owns, and nothing it owns
// itself, so it is a tuple that borrows `v` (an unboxed borrowing record), and a borrow cannot be
// stored into the array. (Before spec 36: an object literal that holds a value it owns, spec 21.2.)
class V {
    constructor(public x: number) {}
}

function main() {
    let v = new V(1);
    const r = { v: v };
    const a: { v: V }[] = [];
    a.push(r);
    print(a[0].v.x);
}

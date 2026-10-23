// -mm=own rejects (spec 21.2): `r` holds the value of the `let` `v`, which `v` owns, and is not seen as
// an interface (where it would be a borrowing record, spec 32), so it moves only into a local or the
// result, not into the array.
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

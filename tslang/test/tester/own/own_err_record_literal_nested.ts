// -mm=own rejects (spec 21.2): `r` holds a literal nested by value, which it does not own outright, so
// it moves only into a local or the result, not into the array. `{ v: new V(1) }` alone moves.
class V {
    constructor(public x: number) {}
}

function main() {
    const r = { inner: { v: new V(1) } };
    const a: { inner: { v: V } }[] = [];
    a.push(r);
    print(a[0].inner.v.x);
}

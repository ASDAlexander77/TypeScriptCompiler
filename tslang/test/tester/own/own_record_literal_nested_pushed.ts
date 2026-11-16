// -mm=own: a record holding a literal nested in it (`{ inner: { v: new V(i) } }`) owns it whole: the
// inner record moves into the outer, and the outer into the array it is pushed to, which frees both.
class V {
    constructor(public x: number) {}
}

function main() {
    let t: number = 0;
    for (let i = 0; i < 2000; i++) {
        const r = { inner: { v: new V(i) } };
        const a: { inner: { v: V } }[] = [];
        a.push(r);
        t += a[0].inner.v.x;
    }

    assert(t == 1999000, "t");
    print("done.");
}

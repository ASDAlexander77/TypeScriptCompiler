// -mm=own, phase 1 rejects: `b` is a later read of `a`'s slot, and `a` moves into the field in
// between. Overwriting the field destroys the value `b` still reads.
class C {
    x: number;
    constructor(x: number) { this.x = x; }
}
class H {
    c: C | null = null;
}
function main() {
    const h = new H();
    let a = new C(5);
    const b = a;
    h.c = a;
    h.c = null;
    print(b.x);
    print("done.");
}

// -mm=own rejects: a function that assigns a parameter keeps it - its local owns the caller's value
// until it is given another - so the caller gave `c` up at the call and may not read it after.
class C {
    constructor(public x: number) {}
}
function reset(c: C): number {
    c = new C(2);
    return c.x;
}
function main() {
    const c = new C(1);
    print(reset(c), c.x);
}

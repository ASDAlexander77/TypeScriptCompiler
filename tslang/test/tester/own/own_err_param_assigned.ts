// -mm=own rejects: a parameter's value is its caller's; this function cannot put a value of its
// own in its place.
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

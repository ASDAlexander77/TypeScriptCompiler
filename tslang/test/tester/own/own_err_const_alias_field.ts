// -mm=own rejects: `b` borrows `a` (a const of a `let` has a slot of its own, #454), and `a`
// moves into the field while `b` still borrows it. Overwriting the field destroys the value `b`
// still reads.
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

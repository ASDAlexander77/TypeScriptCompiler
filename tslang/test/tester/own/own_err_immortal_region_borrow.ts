// -mm=own rejects: `c` is never assigned, but its initializer is a call's result, a borrow of an
// element `list` owns. Marking that block immortal would make the array's free of it a silent
// leak, so `c` stays mortal, and returning it is the error a global's read always was.
class V {
    constructor(public x: number) {}
}
const list = [new V(1)];
function firstOf(a: V[]) {
    return a[0];
}
const c = firstOf(list);
function f() {
    return c;
}
function main() {
    print(f().x);
}

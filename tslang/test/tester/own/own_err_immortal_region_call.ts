// -mm=own rejects: `c` is never assigned, but its initializer is a call's result, a borrow of a
// temporary its region frees. Such a global is not immortal: no call's result is taken as the
// global's own. A global cannot hold a borrow: the initializer is the error (#638).
class V {
    constructor(public x: number) {}
}
function id(v: V) {
    return v;
}
const c = id(new V(1));
function f() {
    return c;
}
function main() {
    print(f().x);
}

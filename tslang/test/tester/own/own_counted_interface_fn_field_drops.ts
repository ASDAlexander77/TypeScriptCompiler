// Accepted under -mm=own since TSLANG 219 (USER RULING 11-27): a local borrowing a place, used after a call that
// may free what it borrows, takes a reference of its own (a copy of a string or an `any`, else a count). It was an
// error; below is what it was refused for.
// -mm=own rejects: a function-typed field of a closed interface whose function overwrites a field of
// the object it is called on. The call through the interface gets that function's facts (spec 31),
// and they say it may destroy what `v` borrows.
class V {
    constructor(public x: number) {}
}
interface S { f: (x: number) => number; v: V; }
class P { s: S = { f: function (x: number) { this.v = new V(9); return x; }, v: new V(5) }; }
function use(p: P) {
    const v = p.s.v;
    p.s.f(1);
    return v.x;
}
function main() {
    print(use(new P()));
}

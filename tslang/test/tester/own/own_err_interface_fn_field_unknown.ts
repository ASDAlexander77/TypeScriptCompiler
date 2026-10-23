// -mm=own rejects: a function-typed field of an interface given a function of unknown origin (a
// parameter). The call through the field cannot be resolved, so it may destroy anything, and the
// borrow of `p.s` it is made through ends at it (spec 31).
interface S { f: (x: number) => number; r: number; }
function ten(x: number) { return x * 10; }
function mkS(fn: (x: number) => number): S { return { f: fn, r: 3 }; }
class P { s: S = mkS(ten); }
function h(p: P) { return p.s.f(3); }
function main() {
    print(h(new P()));
}

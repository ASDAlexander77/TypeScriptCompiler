// -mm=own rejects: a function-typed field of an exported interface. Another module may implement
// it with a function this module never sees, so the call through the field cannot be resolved and
// the borrow of `p.s` it is made through ends at it (spec 31).
export interface S { f: (x: number) => number; r: number; }
class P { s: S = { f: function (x: number) { return x + 1; }, r: 3 }; }
function h(p: P) { return p.s.f(3); }
function main() {
    print(h(new P()));
}

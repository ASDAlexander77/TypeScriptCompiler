// -mm=own rejects: a method whose result borrows a field of `this`, reached through an interface.
// The call through the interface is not given the method's facts, so the caller would release the
// borrow as its own (a double free). Until facts reach interface calls (G9), the method's body is
// the error.
class V {
    constructor(public x: number) {}
}
interface Holder { hold(): V; }
class HH implements Holder { v: V = new V(4); hold(): V { return this.v; } }
function use(h: Holder) { const x = h.hold(); return x.x; }
function main() {
    const hh = new HH();
    print(use(hh), hh.v.x);
}

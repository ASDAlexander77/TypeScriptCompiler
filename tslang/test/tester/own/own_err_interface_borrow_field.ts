// -mm=own rejects: a method whose result borrows a field of `this`, reached through an exported
// interface. Another module may call through the interface without the method's facts, and would
// release the borrow as its own (a double free), so the call is open and the method's body is the
// error. The same program with the interface not exported compiles (own_interface_facts, spec 31).
class V {
    constructor(public x: number) {}
}
export interface Holder { hold(): V; }
class HH implements Holder { v: V = new V(4); hold(): V { return this.v; } }
function use(h: Holder) { const x = h.hold(); return x.x; }
function main() {
    const hh = new HH();
    print(use(hh), hh.v.x);
}

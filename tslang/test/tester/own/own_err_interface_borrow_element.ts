// -mm=own rejects: a method whose result borrows an element of `this`, reached through an exported
// interface. Another module may call through the interface without the method's facts, and would
// release the borrow as its own (a double free), so the call is open and the method's body is the
// error. The same program with the interface not exported compiles (own_interface_facts, spec 31).
class V {
    constructor(public x: number) {}
}
export interface First { first(): V; }
class F implements First { items: V[] = [new V(3)]; first(): V { return this.items[0]; } }
function main() {
    const f = new F();
    const i: First = f;
    const x = i.first();
    print(x.x, f.items[0].x);
}

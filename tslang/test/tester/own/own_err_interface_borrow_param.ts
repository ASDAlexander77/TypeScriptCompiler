// -mm=own rejects: a method whose result borrows its argument, reached through an exported
// interface. Another module may call through the interface without the method's facts, and would
// release the borrow as its own (a double free), so the call is open and the method's body is the
// error. The same program with the interface not exported compiles (own_interface_facts, spec 31).
class V {
    constructor(public x: number) {}
}
export interface Getter { get(v: V): V; }
class G implements Getter { get(v: V): V { return v; } }
function main() {
    const g: Getter = new G();
    const a = new V(5);
    const b = g.get(a);
    print(b.x, a.x);
}

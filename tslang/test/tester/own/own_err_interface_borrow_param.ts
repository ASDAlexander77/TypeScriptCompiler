// -mm=own rejects: a method whose result borrows its argument, reached through an interface. The
// call through the interface is not given the method's facts, so the caller would release the
// borrow as its own (a double free). Until facts reach interface calls (G9), the method's body is
// the error.
class V {
    constructor(public x: number) {}
}
interface Getter { get(v: V): V; }
class G implements Getter { get(v: V): V { return v; } }
function main() {
    const g: Getter = new G();
    const a = new V(5);
    const b = g.get(a);
    print(b.x, a.x);
}

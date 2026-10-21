// -mm=own rejects: a method whose result borrows an element of `this`, reached through an
// interface. The call through the interface is not given the method's facts, so the caller would
// release the borrow as its own (a double free). Until facts reach interface calls (G9), the
// method's body is the error.
class V {
    constructor(public x: number) {}
}
interface First { first(): V; }
class F implements First { items: V[] = [new V(3)]; first(): V { return this.items[0]; } }
function main() {
    const f = new F();
    const i: First = f;
    const x = i.first();
    print(x.x, f.items[0].x);
}

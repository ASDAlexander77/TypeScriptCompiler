// -mm=own rejects: the result of a call through a closed interface borrows an element of the object
// behind the interface value, and the local holding that value is given a new one before the read.
class V {
    constructor(public x: number) {}
}
interface First { first(): V; }
class F implements First { items: V[] = [new V(3)]; first(): V { return this.items[0]; } }
function main() {
    let i: First = new F();
    const x = i.first();
    i = new F();
    print(x.x);
}

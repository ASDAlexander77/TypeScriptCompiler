// -mm=own rejects, with a message that says what to do: a method read through an interface is bound to
// whatever object the interface shows, which a method value cannot own (a class's method value can).
interface HasFn {
    fn: () => int;
}

class C {
    constructor(public v: int) {}
    fn(): int {
        return this.v;
    }
}

function viaInterface(o: HasFn): () => int {
    return o.fn;
}

function main() {
    const c = new C(4);
    print(viaInterface(c)());
}

// -mm=own rejects: a read through an interface's field that a class fills with a getter, returned. On
// the getter's path the read is a new value only the read's own release frees; returned, it would be a
// borrow of `i` to the caller, and leak (one Box per call). A field no class fills with accessors is a
// borrow on every path, and may be returned.
class Box {
    constructor(public n: number) {}
}

interface I {
    item: Box;
}

class Fresh implements I {
    get item() { return new Box(1); }
    set item(x: Box) {}
}

function pick(i: I) {
    return i.item;
}

function main() {
    const f: I = new Fresh();
    print(pick(f).n);
}

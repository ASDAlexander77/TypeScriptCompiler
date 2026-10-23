// An object literal that holds a new value, given as an interface, is boxed into a block of its
// own: the new value moves into the record, the record into the box, and the box is returned
// (spec 27; -mm=own rejected it before).
class Vec {
    constructor(public n: number) {}
}
interface Holder {
    v: Vec;
}
function make(n: number): Holder {
    return { v: new Vec(n) };
}
function main() {
    print(make(1).v.n);
    print("done.");
}

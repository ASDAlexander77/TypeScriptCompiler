// -mm=own rejects: an object literal that holds a new value is moved only into a local or returned
// as it is; given as an interface, it is boxed into a block of its own.
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
}

// -mm=own rejects: `identity` returns its parameter, a result its callers would have to know
// borrows the argument, but an exported function can be called from another module, which does not.
// (A string there is copied instead, spec 22.)
class C {
    constructor(public n: string) {}
}
export function identity(x: C) {
    return x;
}
function main() {
    print(identity(new C("x")).n);
}

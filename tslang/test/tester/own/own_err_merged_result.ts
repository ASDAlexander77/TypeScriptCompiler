// -mm=own rejects: the result of a `?:` is either branch's value, and nothing says which one
// owns it. (A string there is copied instead, spec 22.)
class C {
    constructor(public n: string) {}
}
function max(a: C, b: C) {
    return a.n > b.n ? a : b;
}
function main() {
    print(max(new C("a"), new C("b")).n);
}

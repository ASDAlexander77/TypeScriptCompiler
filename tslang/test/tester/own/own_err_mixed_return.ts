// -mm=own rejects: `pick` returns its argument on one path and a new object on the other, so
// its callers cannot be told whether they own the result. (A string there is copied instead,
// spec 22.)
class C {
    constructor(public n: string) {}
}
function pick(c: C, other: boolean) {
    if (!other) return c;
    return new C("new");
}
function main() {
    const c = new C("a");
    print(pick(c, true).n);
}

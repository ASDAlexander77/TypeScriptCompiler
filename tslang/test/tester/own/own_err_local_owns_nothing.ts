// -mm=own rejects: a local declared without a value owns nothing (MLIRGen makes no owner of it),
// so a new value stored into it later would have no owner to release it.
class C {
    constructor(public x: number) {}
}
function main() {
    let d: C;
    d = new C(1);
    print(d.x);
}

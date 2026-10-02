// -mm=own rejects: a global owns its value, and a function that returns it gives its caller a
// second owner.
class V {
    constructor(public x: number) {}
    static def = new V(1);
}
function getDef() {
    return V.def;
}
function main() {
    print(getDef().x);
}

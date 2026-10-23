// -mm=own rejects: a global owns its value, and a function that returns it gives its caller a
// second owner. `reset` gives the global another value, so it is not immortal: the old value is
// destroyed then, and a caller holding it would hold a freed block.
class V {
    constructor(public x: number) {}
    static def = new V(1);
}
function getDef() {
    return V.def;
}
function reset() {
    V.def = new V(2);
}
function main() {
    print(getDef().x);
    reset();
}

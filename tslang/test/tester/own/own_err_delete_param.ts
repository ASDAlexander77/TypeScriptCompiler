// -mm=own rejects: `delete` ends an owner, and a parameter's value is its caller's - the caller
// would free it again.
class N {
    v = 1;
}
function drop(p: N) {
    delete p;
}
function main() {
    const n = new N();
    drop(n);
    print(n.v);
}

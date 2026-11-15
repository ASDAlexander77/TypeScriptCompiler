// -mm=own rejects: `delete h` ends `h`, and `f` still borrows a field of it.
class C {
    v = 1;
}
class H {
    child = new C();
}
function main() {
    let h = new H();
    const f = h.child;
    delete h;
    print(f.v);
}

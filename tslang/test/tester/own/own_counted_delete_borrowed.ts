// Accepted under -mm=own since TSLANG 221 (USER RULING 11-27, as 219): a local borrowing a place, used after the
// place may be freed - by this function itself, or with the local assigned again - takes a reference of its own
// (a copy of a string or an `any`, else a count). It was an error; below is what it was refused for.
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

// -mm=own rejects: `K.first` is never assigned, but its initializer borrows an element of
// `K.list`, which owns it. Marking that block immortal would make the array's later free of it a
// silent leak, so `K.first` stays mortal and its initializer takes a second reference.
class V {
    constructor(public x: number) {}
}
class K {
    static list = [new V(1)];
    static first = K.list[0];
}
function main() {
    print(K.first.x);
}

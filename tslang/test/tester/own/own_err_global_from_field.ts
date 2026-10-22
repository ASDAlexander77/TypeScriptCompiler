// -mm=own rejects: `k` would be a second owner of the block the field `h.c` owns, and the field's
// overwrite frees what `k` still reads. A global cannot borrow.
class C {
    constructor(public x: number) {}
}

class H {
    c = new C(1);
}

const h = new H();
const k = h.c;

function main() {
    h.c = new C(2);
    print(k.x);
}

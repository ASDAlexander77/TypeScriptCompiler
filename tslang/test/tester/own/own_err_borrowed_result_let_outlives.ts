// -mm=own, phase 4 rejects: `b` is declared from a method's borrowed result, then the owner of
// the field it borrows is assigned.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);
    first() {
        return this.c;
    }
}

function main() {
    let h = new H();
    let b = h.first();
    h = new H();
    print(b.x);
}

// -mm=own, phase 4 rejects: `firstOf(h)` borrows `h`, so it cannot be stored into another
// object, which would outlive `h`'s hold on it.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);
}

function firstOf(h: H) {
    return h.c;
}

function main() {
    const h = new H();
    const h2 = new H();
    h2.c = firstOf(h);
    print(h2.c.x);
}

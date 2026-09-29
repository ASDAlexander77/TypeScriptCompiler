// -mm=own, phase 4 rejects: `c` borrows what `firstOf(h)` returned, a field under `h`; the
// overwrite of `h.c` destroys it before `c.x` reads it.
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
    const c = firstOf(h);
    h.c = new C(9);
    print(c.x);
}

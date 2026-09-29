// -mm=own, phase 4 rejects: `use` is given `h.c`, then assigns the global that owns it and reads
// its parameter. A call given a borrow uses it for as long as the callee runs, so a callee that
// may destroy it is a use after the destroy.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);
}

let g = new H();

function use(c: C) {
    g.c = new C(9);
    print(c.x);
}

function main() {
    const h = g;
    use(h.c);
}

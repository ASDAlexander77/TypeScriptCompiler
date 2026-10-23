// -mm=own rejects: `keep` stores its parameter `c` into a field, and `main` lends it the global `g`,
// which `reset` gives another value elsewhere: `g` is not immortal (spec 25), and a call that may
// drop cannot be lent a value read out of it.
class C {
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

let g = new C(1);

function keep(h: H, c: C) {
    h.c = c;
}

function reset() {
    g = new C(2);
}

function main() {
    const h = new H();
    keep(h, g);
    print(h.c!.x);
    reset();
}

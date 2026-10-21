// -mm=own, phase 4 rejects: `keep` keeps its parameter, but `main` passes a global, which it does
// not own and cannot move. `reset` gives the global another value, so it is not immortal (spec 25).
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

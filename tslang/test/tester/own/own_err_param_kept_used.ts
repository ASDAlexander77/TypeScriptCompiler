// -mm=own, phase 4 rejects: `keep` keeps its parameter, so `a` moves into the call and cannot be
// read after it.
class C {
    constructor(public x: number) {}
}

class H {
    c: C | null = null;
}

function keep(h: H, c: C) {
    h.c = c;
}

function main() {
    const h = new H();
    let a = new C(1);
    keep(h, a);
    print(a.x);
}

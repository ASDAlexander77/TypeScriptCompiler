// -mm=own, phase 4 rejects: `firstOf` is passed as a value, so `apply` calls it where -mm=own
// cannot see which function it is, and would take the result as its own and release it.
// `firstOf`'s body keeps rc's convention and fails, saying why.
class C {
    constructor(public x: number) {}
}

class H {
    c: C = new C(0);
}

function firstOf(h: H) {
    return h.c;
}

function apply(g: (h: H) => C, h: H) {
    return g(h).x;
}

function main() {
    const h = new H();
    print(apply(firstOf, h));
}

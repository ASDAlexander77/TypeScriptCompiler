// -mm=own, phase 5 rejects: `y` borrows `h.c`, and the closure, which holds `h`, overwrites it.
class C {
    constructor(public x: number) {}
}

class H {
    constructor(public c: C) {}
}

function main() {
    const h = new H(new C(1));
    let k = 0;
    const f = () => {
        h.c = new C(2);
        k++;
    };
    const y = h.c;
    f();
    print(y.x, k);
}

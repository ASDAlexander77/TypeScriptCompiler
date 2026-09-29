// -mm=own, phase 4 rejects: `g.c.m()` calls `m` on the borrow of `g.c` itself. `m` assigns
// `g.c`, which destroys its own `this`, and then reads `this`.
class C {
    constructor(public x: number) {}
    m() {
        g.c = new C(9);
        print(this.x);
    }
}

class H {
    c: C = new C(0);
}

let g = new H();

function main() {
    const h = g;
    h.c.m();
}

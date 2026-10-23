// -mm=own: a C function declared without `@nodrops` whose parameters and result are numbers - the
// ray tracer's `sqrt` and `floor` - is given no block of this module and destroys none (spec 31), so
// a borrow of a field survives a call that reaches it: directly, through a function, and through a
// method called through an interface closed in this module. A declaration that takes a block still
// may destroy anything (own_err_declared_drops).
declare function sqrt(v: number): number;
declare function floor(v: number): number;

class V {
    constructor(public x: number) {}
}

class H {
    c: V = new V(16);
}

class M {
    static root(v: number): number {
        return floor(sqrt(v));
    }
}

interface Shape {
    size(h: H): number;
}

class Sq implements Shape {
    size(h: H): number {
        return M.root(h.c.x);
    }
}

function churn() {
    const keep: V[] = [];
    for (let i = 0; i < 2000; i++) {
        keep.push(new V(77777));
    }

    return keep.length;
}

function direct(h: H) {
    const c = h.c;
    const r = sqrt(c.x);
    churn();
    return r + c.x;
}

function viaMethod(h: H) {
    const c = h.c;
    const r = M.root(c.x + 9);
    churn();
    return r + c.x;
}

function viaInterface(s: Shape, h: H) {
    const c = h.c;
    const r = s.size(h);
    churn();
    return r + c.x;
}

function main() {
    const h = new H();
    assert(direct(h) == 20, "across sqrt");
    assert(viaMethod(h) == 21, "across a function that calls sqrt and floor");
    assert(viaInterface(new Sq(), h) == 20, "across a call through an interface");
    print(direct(h), viaMethod(h), viaInterface(new Sq(), h));   // 20 21 20
    print("done.");
}

// A function value boxed into `any` and cast back to a function type (`f as any as (...) => R`).
// A plain function is boxed as its pointer alone, a closure or a bound method as { function, this };
// cast back, the plain one read past its box and took what followed for `this`, which only the
// collector's zeroing allocator made null: under rc, none and own the call went through garbage.
class V {
    constructor(public x: number) {}
}

interface Op {
    run(v: V): number;
}

interface Op2 {
    run(v: V): number;
}

class OB implements Op {
    run(v: V): number {
        return v.x + 1;
    }
}

class C {
    k = 10;
    m(a: number) {
        return a + this.k;
    }
}

function add(a: number, b: number) {
    return a + b;
}

function x(v: V) {
    return v.x;
}

function viaOp2(o: Op2, v: V) {
    return o.run(v);
}

function main() {
    const fadd = add as any as ((a: number, b: number) => number);
    assert(fadd(1, 2) == 3, "plain function");

    const fx = x as any as ((v: V) => number);
    assert(fx(new V(4)) == 4, "class parameter");

    const fop = viaOp2 as any as ((o: Op, v: V) => number);
    assert(fop(new OB(), new V(5)) == 6, "interface parameter");

    let base = 100;
    const cl = (a: number) => a + base;
    const fcl = cl as any as ((a: number) => number);
    assert(fcl(1) == 101, "closure");

    const c = new C();
    const bm = c.m as any as ((a: number) => number);
    assert(bm(2) == 12, "bound method");

    for (let i = 0; i < 1000; i++) {
        const f = add as any as ((a: number, b: number) => number);
        assert(f(i, 1) == i + 1, "in a loop");
    }

    print("done.");
}

// a class's `v?: T` is `T | undefined`: unassigned it reads as undefined, directly and through an
// interface's `v?: T` (whose slot then reads the field's flag); a write through the interface sets it
interface I {
    v?: string;
    n?: number;
    k: number;
}

class C implements I {
    v?: string;
    n?: number;
    constructor(public k: number) {}
}

class Base {
    a = 1;
    v?: string;
}

class D extends Base implements I {
    k: number = 2;
    x?: boolean = true;
}

class G<T> {
    g?: T;
}

interface J {
    u?: number;
}

class E implements J {
    pad = "p";
    u?: number;
}

class P {
    constructor(public m?: string, public n?: number) {}
}

type L = { u?: number };

function viaInterface(i: I) {
    return i.n ?? -1;
}

function sum(j: J) {
    let s = 0;
    for (let k = 0; k < 3; k++) {
        if (j.u === undefined) {
            j.u = 10;
        }

        s += j.u ?? 100;
    }

    return s;
}

function viaLiteral(l: L) {
    return l.u ?? -1;
}

function main() {
    const c = new C(1);
    assert(c.v === undefined && c.n === undefined, "unassigned");
    assert((c.n ?? -1) == -1, "unassigned ??");
    assert(viaInterface(c) == -1, "unassigned through an interface");

    c.n = 0;
    assert(c.n === 0, "assigned");
    assert(viaInterface(c) == 0, "zero through an interface");

    const i: I = c;
    assert(i.v === undefined, "unassigned string through an interface");
    i.v = "x";
    assert(c.v == "x" && i.v == "x", "written through an interface");
    i.n = 5;
    assert(c.n == 5 && i.n == 5, "number written through an interface");
    i.v = undefined;
    assert(c.v === undefined && i.v === undefined, "undefined written through an interface");
    c.n = undefined;
    assert(i.n === undefined, "undefined written directly");

    const d = new D();
    const di: I = d;
    assert(di.v === undefined, "inherited field");
    di.v = "dv";
    assert(d.v == "dv" && di.v == "dv", "inherited field written through an interface");
    assert(d.x === true, "initialized");
    d.x = undefined;
    assert(d.x === undefined, "cleared");

    const g = new G<number>();
    assert(g.g === undefined, "generic");
    g.g = 3;
    assert(g.g == 3, "generic assigned");

    const e = new E();
    assert(sum(e) == 30 && e.u == 10, "loop through an interface");

    const e2 = new E();
    assert(viaLiteral(e2) == -1, "to a type literal");
    e2.u = 4;
    assert(viaLiteral(e2) == 4, "to a type literal, assigned");

    const p = new P();
    assert(p.m === undefined && p.n === undefined, "constructor parameter fields");
    const p2 = new P("a", 0);
    assert(p2.m == "a" && p2.n === 0, "constructor parameter fields, given");

    const literal: I = { k: 3 };
    assert(literal.v === undefined, "object literal without the field");

    print("done.");
}

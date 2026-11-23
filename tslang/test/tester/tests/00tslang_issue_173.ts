// TSLANG 173: a generic function given to a function type whose parameter is `any` (a typed `const` holding a generic
// arrow) infers its type parameter as `any` (it was "can't find type by name: p"); a namespace `var` declared in a
// block of the namespace's code is seen by the namespace's functions (it was "can't resolve name"); a class can
// extend a module `const` set from a class

interface W<p> {
    a: p;
}

const f: <p>(a: p) => W<p> = <p>(a: p): W<p> => ({ a });

function id<p>(a: p): p {
    return a;
}

const g: (a: any) => any = id;

namespace N {
    for (var i = 0; i < 2; i++) {}
    for (var i = 0; i < 3; i++) {}

    if (i > 0) {
        var k = 10;
    }

    export function get() {
        return i + k;
    }
}

class B {
    v = 1;
    hi() {
        return "B" + this.v;
    }
}

const base = B;

class D extends base {
    w = 2;
    hi() {
        return "D" + this.w + super.hi();
    }
}

function main() {
    const w = f(5);
    assert(w.a == 5);

    assert(g(3) == 3);

    assert(N.get() == 13);

    const d = new D();
    const b: B = d;
    assert(d.v == 1 && d.w == 2);
    assert(b.hi() == "D2B1");
    assert(d instanceof B);

    print("done.");
}

// `x instanceof C` where `x` is a union or an optional that may hold a class: whether it holds one
// is known only at run time, and which class only to the instance's vtable. It used to fold to the
// compile-time question "is the union type C", which is false, so every one of these was false.
class C {
    constructor(public x: number) {}
}

class Sub extends C {
    constructor(x: number) {
        super(x);
    }
}

class D {
    y = 1;
}

function isC(v: C | D) {
    return v instanceof C;
}

function main() {
    // a nullable class: a pointer that may be null
    let e: C | null = new C(1);
    assert(e instanceof C, "C | null holding a C");
    if (e) assert(e instanceof C, "narrowed C | null");
    e = null;
    assert(!(e instanceof C), "C | null holding null");

    // a subclass held as its base, nullable
    let s: C | null = new Sub(2);
    assert(s instanceof C && s instanceof Sub, "C | null holding a Sub");

    // a union of classes: tagged, every class member's tag is "class"
    let b: C | D = new C(3);
    assert(b instanceof C && !(b instanceof D), "C | D holding a C");
    b = new D();
    assert(b instanceof D && !(b instanceof C), "C | D holding a D");
    assert(isC(new C(4)) && !isC(new D()), "C | D parameter");

    // a class and a number
    let f: C | number = new C(5);
    assert(f instanceof C, "C | number holding a C");
    f = 6;
    assert(!(f instanceof C), "C | number holding a number");

    // an optional class
    let o: C | undefined = new C(7);
    assert(o instanceof C, "C | undefined holding a C");
    o = undefined;
    assert(!(o instanceof C), "C | undefined holding undefined");

    // a plain class still asks its own vtable
    const c = new C(8);
    assert(c instanceof C && !(c instanceof Sub), "plain C");

    print("done.");
}

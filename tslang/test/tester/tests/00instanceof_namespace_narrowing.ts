// `if (x instanceof N.C)` narrows x to N.C inside the if - and crashed the compiler: the
// narrowing built a type reference out of `N.C` as the property access it is, and a type
// reference names its type with an entity name (an identifier or a qualified name).

namespace N {
    export class C {
        code = 5;
    }

    export namespace Inner {
        export class D {
            code = 7;
        }
    }
}

function main() {
    let a: any = new N.C();
    let r = 0;
    if (a instanceof N.C) {
        r = a.code;
    }

    assert(r == 5, "narrowed to N.C");

    let b: any = new N.Inner.D();
    let s = 0;
    if (b instanceof N.Inner.D) {
        s = b.code;
    }

    assert(s == 7, "narrowed to N.Inner.D");

    let t = 0;
    if (b instanceof N.C) {
        t = 1;
    }

    assert(t == 0, "an N.Inner.D is not an N.C");

    print("done.");
}

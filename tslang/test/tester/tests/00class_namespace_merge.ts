// A class and a namespace of its name (`class C {} namespace C { export var x }`): the name is the class, and a
// member it hasn't got is the namespace's (`C.x`). As the namespace, `new C()` was "'C' is a namespace, not a class".
class C {
    constructor(public v: number) {}

    static boo() {
        return 2;
    }
}

namespace C {
    export var x = 1;
    export function make() {
        return new C(5);
    }
}

namespace C {
    export const label = "c";
}

namespace N {
    export class D {
        w = 3;
    }

    export namespace D {
        export var y = 12;
    }
}

function main() {
    const c = new C(C.x);
    assert(c.v == 1);
    assert(C.boo() == 2);
    assert(C.make().v == 5);
    assert(C.label == "c");
    assert(new N.D().w == 3 && N.D.y == 12);
    print("done.");
}

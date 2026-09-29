// -mm=rc: a call through an interface hands back the +1 its implementation retained, and the
// caller takes it over instead of keeping it. Which implementations can be in the slot is read
// off the vtable globals, whose slot 0 is the `.instanceOf` header: `make` is member 0 and sits
// in slot 1. Read at the member's own index, `i.make()` asked `.instanceOf` - no heap result, so
// unclassified - and leaked every C it made.
//
// The generator matters: its `next` returns without retaining, which turns off the whole-module
// shortcut that would otherwise let every indirect call consume, and so hide the lookup.
class C {
    v: number[] = [];
    constructor(public x: number) {}
}

interface I {
    make(): C;
    other(): C;
}

class B implements I {
    make() {
        return new C(1);
    }

    other() {
        return new C(2);
    }
}

function* gen() {
    yield new C(3);
}

function main() {
    let t: number = 0;
    const i: I = new B();
    for (let k = 0; k < 1000; k++) {
        const c = i.make();
        const d = i.other();
        t += c.x + d.x;
    }

    for (const g of gen()) t += g.x;

    assert(t == 3003);
    print("done.");
}

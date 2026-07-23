// Regression test for the accessor-vtable-dispatch-fix: overriding a
// get/set accessor pair in a subclass and accessing it through a
// BASE-TYPED reference (an ordinary upcast, not `super`) used to always
// resolve statically to the base's own accessor - accessors were never
// part of the vtable dispatch decision at all, unlike ordinary methods.
class Base {
    protected _val: number = 10;

    get val(): number {
        return this._val;
    }

    set val(v: number) {
        this._val = v;
    }
}

class Derived extends Base {
    get val(): number {
        return this._val * 2;
    }

    set val(v: number) {
        this._val = v + 1;
    }
}

function main() {
    const d = new Derived();
    const asBase: Base = d;

    asBase.val = 5;
    assert(asBase.val == 12);

    // and through the derived-typed reference directly, for good measure
    assert(d.val == 12);

    print("done.");
}

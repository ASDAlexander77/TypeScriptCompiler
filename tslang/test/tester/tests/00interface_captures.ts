// regression test: an object-literal method that captures an outer variable,
// cast to an interface and called through the interface reference. Confirms
// the constant-vtable optimization (see
// docs/interface-vtable-simplification-design.md §3) works correctly for
// captures-bearing methods, not just capture-free ones: the vtable slot gets
// a compile-time-constant SymbolRefOp (the lifted method is per-literal-
// expression, shared by every instance), while each instance's captured data
// lives in a separate per-object `.captured` field the method reads via
// `this` - so two independently-created instances must not share captured
// state.

interface Getter {
    get(): number;
}

function make(x: number): Getter {
    return {
        get() {
            return x;
        }
    };
}

function main() {
    const g1: Getter = make(42);
    const g2: Getter = make(100);

    assert(g1.get() == 42);
    assert(g2.get() == 100);
    // the two instances must not share captured state
    assert(g1.get() == 42);

    print("done.");
}

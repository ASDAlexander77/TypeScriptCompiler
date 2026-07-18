// regression test: a top-level (global) binding whose declared type is an interface,
// initialized from an object literal that has a method, used to crash the JIT with
// 0xC0000005 on the first field/method access.
// Root cause: casting the (boxed) object literal to the interface builds a per-object
// vtable patched with the method's function pointer; that vtable was a stack VariableOp
// (alloca). For a local binding the alloca outlives its uses, but a global binding's
// initializer lowers to a __cctor function, so the alloca dangled once the __cctor
// returned -- the interface's vtable pointer then referenced freed stack memory.
// Fix: heap-allocate the patched vtable (mlirGenCreateInterfaceVTableForObject), same
// footing as the object it describes.

interface Counter {
    count: number;
    inc(): void;
}

const counter: Counter = {
    count: 0,
    inc() { this.count = this.count + 1; },
};

// a second same-shape global to make sure two independent per-object vtables both survive
const other: Counter = {
    count: 100,
    inc() { this.count = this.count + 10; },
};

function main() {
    assert(counter.count == 0);
    counter.inc();
    counter.inc();
    assert(counter.count == 2);

    other.inc();
    assert(other.count == 110);
    // the two globals must not share state
    assert(counter.count == 2);

    print("done.");
}

// The atomic builtins: atomicrmw, cmpxchg, atomicload, atomicstore and fence, with their optional
// syncscope, alignment and volatile / weak arguments. The value is cast to the type the reference
// points at, so a literal does not make a 16-bit element a 32-bit store. Orderings are LLVM's:
// 2 monotonic, 4 acquire, 5 release, 6 acq_rel, 7 seq_cst.

enum Order {
    Monotonic = 2,
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7
}

class Counter {
    value: i32 = 0;
    other: u16 = 7;
}

function main() {
    // atomicrmw returns the value before the operation
    let a: i32 = 20;
    assert(atomicrmw(1, ReferenceOf(a), 2, Order.SequentiallyConsistent) == 20 && a == 22, "add");
    assert(atomicrmw(2, ReferenceOf(a), 2, Order.Monotonic, "", 4) == 22 && a == 20, "sub, syncscope and alignment");
    assert(atomicrmw(0, ReferenceOf(a), 5, 7, "", 0, true) == 20 && a == 5, "volatile xchg");

    // the literal is cast to u16: the add wraps at 16 bits and the neighbour is untouched
    const c = new Counter();
    c.other = 65535;
    assert(atomicrmw(1, ReferenceOf(c.other), 1, 7) == 65535 && c.other == 0, "u16 wraps");
    assert(c.value == 0, "the field before it untouched");

    let f: number = 1.5;
    assert(atomicrmw(11, ReferenceOf(f), 2, 7) == 1.5 && f == 3.5, "fadd");

    // atomicload / atomicstore
    let b: i64 = 5;
    assert(atomicload(ReferenceOf(b), Order.Acquire) == 5, "load");
    atomicstore(ReferenceOf(b), 9, Order.Release);
    assert(b == 9, "store");
    assert(atomicload(ReferenceOf(b), 7, "", 8, true) == 9, "volatile load");
    atomicstore(ReferenceOf(b), 10, 7, "", 0, true);
    assert(atomicload(ReferenceOf(b), 1) == 10, "volatile store, unordered load");

    let n: number = 0.25;
    atomicstore(ReferenceOf(n), 0.5, 7);
    assert(atomicload(ReferenceOf(n), 7) == 0.5, "a float");

    // cmpxchg returns [the value before, whether it was the expected one]
    let x: i32 = 22;
    const hit = cmpxchg(ReferenceOf(x), 22, 30, 7, 7);
    assert(hit[0] == 22 && hit[1] && x == 30, "cmpxchg hit");
    const miss = cmpxchg(ReferenceOf(x), 22, 40, Order.AcquireRelease, Order.Acquire);
    assert(miss[0] == 30 && !miss[1] && x == 30, "cmpxchg miss");

    // a weak cmpxchg may fail spuriously: retried until it succeeds
    let tries = 0;
    let done = false;
    while (!done && tries < 1000) {
        const weak = cmpxchg(ReferenceOf(x), 30, 31, 7, 7, "", 0, true, true);
        done = weak[1];
        tries++;
    }

    assert(done && x == 31, "weak volatile cmpxchg");

    fence(Order.SequentiallyConsistent);
    fence(4, "");

    print("done.");
}

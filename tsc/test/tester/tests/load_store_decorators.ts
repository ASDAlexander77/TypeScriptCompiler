enum MemoryOrder {
    NotAtomic = 0,
    Unordered = 1,
    Monotonic = 2,
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}

// load - acquire, store - release
@atomic(MemoryOrder.Acquire, "test")
@volatile
@nontemporal
@invariant
let a = 1;
a = 2;
print(a);

@atomic(MemoryOrder.AcquireRelease, "test")
@volatile
@nontemporal
@invariant
let b = 3;
b = 4;
print(b);

@atomic(MemoryOrder.SequentiallyConsistent, "test")
@volatile
@nontemporal
@invariant
let c = 5;
c = 6;
print(c);

print("done.");
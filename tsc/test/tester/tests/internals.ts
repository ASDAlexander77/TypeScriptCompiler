// link test
//linker_options("/DEFAULTLIB:libtest");

// inline asm test
const v1: i64 = -42;
const r1 = inline_asm<i64>("xor $0, $0", "=r,r", v1);
assert(r1 == 0)
print(v1, r1);

// call intrinsic test
const v2: f32 = 1.1;
const r2 = call_intrinsic<f32>("llvm.round", v2);
assert(r2 == 1.0)
print(v2, r2);

// // fence test
enum MemoryOrder {
    NotAtomic = 0,
    Unordered = 1,
    Monotonic = 2,
    Acquire = 4,
    Release = 5,
    AcquireRelease = 6,
    SequentiallyConsistent = 7,
}
// fence(MemoryOrder.Acquire);
// fence(MemoryOrder.AcquireRelease, "scope1");

enum AtomicBinOp {
    Xchg = 0,
    Add = 1,
    Sub = 2,
    And = 3,
    Nand = 4,
    Or = 5,
    Xor = 6,
    Max= 7,
    Min= 8,
    UMax= 9,
    UMin= 10,
    FAdd= 11,
    FSub= 12,
    FMax= 13,
    FMin= 14,
    UIncWrap = 15,
    UDecWrap = 16,
}

let a: i32 = 20;
const r3 = atomicrmw(AtomicBinOp.Xchg, ReferenceOf(a), 11, MemoryOrder.Acquire);
assert(r3 == 20 && a == 11)
print(r3, a);

print("done.");
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

// fence test
const 
    not_atomic = 0,
    unordered = 1,
    monotonic = 2,
    acquire = 4,
    release = 5,
    acq_rel = 6,
    seq_cst = 7;
fence(acquire);
fence(acq_rel, "scope1");

print("done.");
// The smallest program that reads a catch variable, and the one that shows item 5am.
//
// DISABLED under the JIT, where it reads uninitialised memory: the same binary run three times
// gives 134, 131, 184 under `gc`, a stable 0 under `rc`, and 73/135/15 under `none`. Ahead of
// time it is correct in every model. Registered and disabled rather than left out, so ctest
// counts it and the build says what is broken - the convention the BROKEN lists exist for.
//
// Size is what decides it, which is why this file is minimal and `00catch_value.ts` is not:
// that one reads six catch values and passes in both tiers, exactly as `00try_catch.ts` does.
// Section 9.29 saw the same thing from the other side and read it as "only in a module that
// throws just that one type"; section 9.65 measured it as a back-end difference instead. The
// LLVM IR is right - the catchpad names the slot, the load reads that slot, `_CT??_R0H@84`
// carries the correct `sizeOrOffset`. What differs is the image base every RVA in those
// descriptors is truncated against, and the JIT reaches `_CxxThrowException` through the shim
// of section 9.13. The handler is found and the clause runs; the object never arrives.

let t = 0;

function main() {
    try {
        throw 2;
    }
    catch (v: TypeOf<1>) {
        t = v;
    }

    assert(t == 2, "a catch clause binds the value that was thrown");

    print("done.");
}

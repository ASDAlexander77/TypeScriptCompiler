// A thrown primitive number has to be catchable by a generic handler, not just by a handler
// declared with the exact matching numeric type.
//
// Root cause: on Windows, an untyped `catch (e)` and a `catch (e: any)` both compile to a
// single SEH catchpad that filters on the same generic `void*`-shaped RTTI descriptor used for
// `any` (`??_R0PEAX@8`). For that filter to match, the thrown value's ThrowInfo has to list a
// compatible `.PEAX` CatchableType entry alongside its own - which is exactly what a thrown
// string or class instance already carries (see StringType::typeName2 / ClassType::typeName2
// in LLVMRTTIHelperVCWin32Const.h). `int`, `float`, and `double` did not carry that second
// entry, so a thrown number's CatchableTypeArray held only its own primitive descriptor; the
// generic filter never found a match, __CxxFrameHandler3 walked past `main` with nothing left
// to catch it, and the process quietly ended instead of running the catch body.
//
// Do NOT assert on the caught value here - see 00catch_value.ts's note (item 5am) on why
// reading a `catch` binding is a separate, already-tracked, back-end-dependent question. This
// file only asserts control flow: the catch body runs, and the statement after the try/catch
// still runs too.

function untypedCatchesInt() {
    let ran = 0;
    try {
        throw 42;
    }
    catch (e) {
        ran = 1;
    }
    ran = ran + 10;
    return ran;
}

function anyCatchesInt() {
    let ran = 0;
    try {
        throw 42;
    }
    catch (e: any) {
        ran = 1;
    }
    ran = ran + 10;
    return ran;
}

function untypedCatchesFloat() {
    let ran = 0;
    try {
        throw 2.5;
    }
    catch (e) {
        ran = 1;
    }
    ran = ran + 10;
    return ran;
}

function anyCatchesFloat() {
    let ran = 0;
    try {
        throw 2.5;
    }
    catch (e: any) {
        ran = 1;
    }
    ran = ran + 10;
    return ran;
}

// a `number`-typed value (not a bare literal) thrown and caught the same way
function untypedCatchesNumberVar() {
    let n: number = 42;
    let ran = 0;
    try {
        throw n;
    }
    catch (e) {
        ran = 1;
    }
    ran = ran + 10;
    return ran;
}

function main() {
    assert(untypedCatchesInt() == 11, "an untyped catch must catch a thrown int and control flow must continue");
    assert(anyCatchesInt() == 11, "a `catch (e: any)` must catch a thrown int and control flow must continue");
    assert(untypedCatchesFloat() == 11, "an untyped catch must catch a thrown float and control flow must continue");
    assert(anyCatchesFloat() == 11, "a `catch (e: any)` must catch a thrown float and control flow must continue");
    assert(untypedCatchesNumberVar() == 11, "an untyped catch must catch a thrown `number` variable and control flow must continue");

    print("done.");
}

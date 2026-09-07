// The smallest program that reads a catch variable, and the one that found item 5am.
//
// It read uninitialised memory under the JIT - the same binary run three times gave 134, 131,
// 184 under `gc`, a stable 0 under `rc`, 73/135/15 under `none` - while being correct ahead of
// time in every model. Section 9.29 saw the same thing from the other side and read it as "only
// in a module that throws just that one type"; the real variable was size, and underneath that,
// which section RTDyld placed lowest.
//
// RTDyld resolves image-relative relocations against the LOWEST section load address, so a datum
// there has RVA 0 - and RVA 0 is the MSVC C++ EH encoding's "none" sentinel. When this file's
// `??_R0H@8` landed at the base, the clause read `dispType == 0`, which is `catch(...)`: it
// still caught, so nothing looked wrong, but a catch-all has no catch object and the value was
// never copied. Fixed by reserving one contiguous block laid out code-first, so the RVA-0
// collision can only ever fall on code, where nothing reads 0 as "none". Sections 9.66 and 9.68.
//
// Kept minimal on purpose: `00catch_value.ts` covers the same feature more broadly and passed
// throughout, because six catch values is enough content to push the descriptor off the base.
// This file is the one that was small enough to break, so it is the one that guards the fix.

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

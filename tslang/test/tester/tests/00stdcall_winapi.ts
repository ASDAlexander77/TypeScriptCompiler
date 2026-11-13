// The Windows API is __stdcall: on 32-bit Windows the callee pops its arguments, and kernel32.lib names
// each function _Name@<bytes of arguments>. `@stdcall` says so of a declaration; on x64 it changes
// nothing. A call with the wrong convention leaves the stack off by the arguments' size every time,
// which a loop of calls turns into a crash.

@stdcall
declare function GetCurrentProcessId(): u32;

@stdcall
declare function MulDiv(number: s32, numerator: s32, denominator: s32): s32;

function main() {
    const id = GetCurrentProcessId();
    assert(id != 0, "a call with no arguments");

    let sum = 0;
    for (let i = 0; i < 1000; i++) {
        sum += MulDiv(i, 6, 3);
    }

    assert(sum == 999000, "calls with arguments, which the callee pops");

    // taken as a value, it is called as any function value is
    const muldiv = MulDiv;
    assert(muldiv(10, 3, 2) == 15, "through a function value");
    const processId = GetCurrentProcessId;
    assert(processId() == id, "the same process");

    print("done.");
}

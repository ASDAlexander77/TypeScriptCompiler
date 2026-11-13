import './export_stdcall'

// `@stdcall` functions of another module: called here as __stdcall, by their decorated names on 32-bit
// Windows, and one given to _beginthreadex as a thread's start.

@cdecl declare function _beginthreadex(security: Opaque | null, stack: u32, start: (arg: Opaque) => u32, arg: Opaque, flags: u32, id: Opaque | null): Opaque;
@stdcall declare function WaitForSingleObject(handle: Opaque, ms: u32): u32;
@stdcall declare function CloseHandle(handle: Opaque): s32;

function main() {
    let sum = 0;
    for (let i = 0; i < 1000; i++) {
        sum += scale(i);
    }

    assert(sum == 1498500, "a direct call, which the callee's convention pops");

    const viaValue = scale;
    assert(viaValue(5) == 15, "through a function value");

    const values: s32[] = [42];
    const thread = _beginthreadex(null, 0, threadStart, <Opaque>Ref(values[0]), 0, null);
    WaitForSingleObject(thread, 0xFFFFFFFF);
    CloseHandle(thread);
    assert(startedWith() == 42, "a thread started with the other module's callback");

    print("done.");
}

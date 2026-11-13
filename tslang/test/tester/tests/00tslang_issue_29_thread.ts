// TSLANG_ISSUES 29 (Windows): a thread started by the C runtime with a @cdecl callback runs the
// TypeScript function it was given, which sees the argument passed through to it.

@cdecl declare function _beginthreadex(security: Opaque | null, stack: u32, start: (arg: Opaque) => u32, arg: Opaque, flags: u32, id: Opaque | null): Opaque;
// the Windows API, and the thread's start, which _beginthreadex calls, are __stdcall (on 32-bit Windows
// that is how they are called and named)
@stdcall declare function WaitForSingleObject(handle: Opaque, ms: u32): u32;
@stdcall declare function CloseHandle(handle: Opaque): s32;

let seen: s32 = 0;

@stdcall
function entry(arg: Opaque): u32 {
    seen = Deref(<Ref<s32>>arg);
    return 0;
}

function main() {
    const values: s32[] = [42];
    const thread = _beginthreadex(null, 0, entry, <Opaque>Ref(values[0]), 0, null);
    WaitForSingleObject(thread, 0xFFFFFFFF);
    CloseHandle(thread);
    assert(seen == 42, "the thread ran the callback with its argument");
    print("done.");
}

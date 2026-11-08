// TSLANG_ISSUES 29 (Windows): a thread started by the C runtime with a @cdecl callback runs the
// TypeScript function it was given, which sees the argument passed through to it.

@cdecl declare function _beginthreadex(security: Opaque | null, stack: u32, start: (arg: Opaque) => u32, arg: Opaque, flags: u32, id: Opaque | null): Opaque;
declare function WaitForSingleObject(handle: Opaque, ms: u32): u32;
declare function CloseHandle(handle: Opaque): s32;

let seen: s32 = 0;

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

// TSLANG_ISSUES 29 (POSIX): a thread started by pthread_create with a @cdecl callback runs the
// TypeScript function it was given, which sees the argument passed through to it.

@cdecl declare function pthread_create(thread: Opaque, attributes: Opaque | null, start: (arg: Opaque) => Opaque, arg: Opaque): s32;
declare function pthread_join(thread: u64, result: Opaque | null): s32;

let seen: s32 = 0;

function entry(arg: Opaque): Opaque {
    seen = Deref(<Ref<s32>>arg);
    return arg;
}

function main() {
    const values: s32[] = [42];
    let thread: u64 = 0;
    assert(pthread_create(<Opaque>Ref(thread), null, entry, <Opaque>Ref(values[0])) == 0, "pthread_create");
    assert(pthread_join(thread, null) == 0, "pthread_join");
    assert(seen == 42, "the thread ran the callback with its argument");
    print("done.");
}

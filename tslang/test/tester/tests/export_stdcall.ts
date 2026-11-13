// The library side of import_stdcall.ts: `@stdcall` functions with a body, exported. On 32-bit Windows
// they are __stdcall and named so (_scale@4, _threadStart@4), and an importer has to call them the same
// way and by the same names: its declaration of them says @stdcall too.

let started: s32 = 0;

export function startedWith(): s32 {
    return started;
}

// called by the importer, directly
@stdcall
export function scale(value: s32): s32 {
    return value * 3;
}

// given by the importer to _beginthreadex, which calls it as __stdcall
@stdcall
export function threadStart(arg: Opaque): u32 {
    started = Deref(<Ref<s32>>arg);
    return 0;
}

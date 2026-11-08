// TSLANG_ISSUES 29: a C callback is a code pointer alone, so a closure - which needs its captured
// variables - can't be one
@cdecl declare function qsort(base: Opaque, count: u64, size: u64, compare: (a: Opaque, b: Opaque) => s32): void;

function main() {
    const items: s32[] = [2, 1];
    let calls = 0;
    qsort(<Opaque>Ref(items[0]), 2, 4, (a: Opaque, b: Opaque) => { calls++; return <s32>0; });
    print("done.");
}

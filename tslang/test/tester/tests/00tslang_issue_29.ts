// TSLANG_ISSUES 29: a parameter of function type in a `declare function` was a TypeScript function
// value - three pointers (code, `this`, captures) - while C takes a callback as one code pointer, so a
// C function given a callback called garbage. A function declared @cdecl takes each callback
// parameter as a plain C function pointer: a function, or an arrow that captures nothing.

@cdecl declare function qsort(base: Opaque, count: u64, size: u64, compare: (a: Opaque, b: Opaque) => s32): void;

function ascending(a: Opaque, b: Opaque): s32 {
    return Deref(<Ref<s32>>a) - Deref(<Ref<s32>>b);
}

function main() {
    const items: s32[] = [5, 3, 9, 1, 7];
    qsort(<Opaque>Ref(items[0]), 5, 4, ascending);
    assert(items[0] == 1 && items[1] == 3 && items[2] == 5 && items[3] == 7 && items[4] == 9, "a function");

    qsort(<Opaque>Ref(items[0]), 5, 4, (a: Opaque, b: Opaque) => <s32>(Deref(<Ref<s32>>b) - Deref(<Ref<s32>>a)));
    assert(items[0] == 9 && items[1] == 7 && items[2] == 5 && items[3] == 3 && items[4] == 1, "an arrow capturing nothing");

    print("done.");
}

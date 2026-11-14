// TSLANG_ISSUES 29: a C comparator given as a function value. A capturing arrow assigned to a type whose
// parameters are `Opaque` (`(a: Opaque, b: Opaque) => s32`) was refused: the leading `Opaque` was taken
// for a receiver. It is a closure now, callable directly; given to qsort through a variable or a
// parameter, a plain function sorts and a closure throws (it has captures the C pointer can't carry).
@cdecl declare function qsort(base: Opaque, count: index, size: index, compare: (a: Opaque, b: Opaque) => s32): void;
type Cmp = (a: Opaque, b: Opaque) => s32;
function byValue(a: Opaque, b: Opaque): s32 { return Deref(<Ref<s32>>a) - Deref(<Ref<s32>>b); }
function sortWith(items: s32[], compare: Cmp): boolean {
    try { qsort(<Opaque>Ref(items[0]), items.length, 4, compare); } catch (e) { return false; }
    return true;
}
function main() {
    const items: s32[] = [3, 1, 2];
    const viaPlain: Cmp = byValue;
    assert(sortWith(items, viaPlain) && items[0] == 1 && items[1] == 2 && items[2] == 3, "a plain function");
    const offset = 0;
    const capturing: Cmp = (a: Opaque, b: Opaque) => <s32>(byValue(b, a) + offset);
    assert(capturing(<Opaque>Ref(items[0]), <Opaque>Ref(items[1])) > 0, "a closure called directly");
    assert(!sortWith(items, capturing), "a closure as a C comparator throws");
    print("done.");
}

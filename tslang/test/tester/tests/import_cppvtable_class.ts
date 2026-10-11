import './export_cppvtable_class'

// The "C++ object" made by hand, as in 00cppvtable_class.ts: a table of functions and a tuple
// whose first field points at it.

type ObjectData = [vtable: Opaque, value: s32];

declare function memcpy(dst: Opaque, src: Opaque, count: index): Opaque;

function slot0(self: Opaque): s32 {
    return 10;
}

function slot1(self: Opaque, x: s32): s32 {
    return Deref(<Ref<ObjectData>>self).value + x;
}

function slot2(self: Opaque, x: s32, y: s32): s32 {
    return x * y;
}

// returns a record as MSVC does: through the pointer after `this`
function slot3(self: Opaque, result: Ref<M.Pair>, x: s32): Ref<M.Pair> {
    let value: M.Pair = [x, x + 1];
    memcpy(<Opaque>result, <Opaque>Ref(value), sizeof<M.Pair>());
    return result;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slot0, <Opaque>slot1, <Opaque>slot2, <Opaque>slot3];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 32];
    const object = <Opaque>Ref(data);

    const derived = <M.Derived>object;
    assert(derived.value == 32, "the first field follows the vtable pointer");
    assert(derived.first() == 10, "slot 0");
    assert(derived.second(5) == 37, "slot 1");
    assert(derived.third(6, 7) == 42, "slot 2");
    assert(M.callSecond(derived, 8) == 40, "the exporting module calls the same slot");
    const pair = derived.pair(7);
    assert(pair.first == 7 && pair.second == 8, "a record returned by a method of an imported class");
    const exportedPair = M.callPair(derived, 9);
    assert(exportedPair.first == 9 && exportedPair.second == 10, "and by the exporting module");

    print("done.");
}

import './export_cppvtable_hidden_return'

// The "C++ object" made by hand, as in 00cppvtable_hidden_return.ts: a table of functions and a
// tuple whose first field points at it.

type ObjectData = [vtable: Opaque, value: s32];

declare function memcpy(dst: Opaque, src: Opaque, count: index): Opaque;

function slot0(self: Opaque): s32 {
    return 10;
}

// returns a record as MSVC does: through the pointer after `this`
function slot1(self: Opaque, result: Ref<M.Pair>, x: s32): Ref<M.Pair> {
    let value: M.Pair = [x + Deref(<Ref<ObjectData>>self).value, x + 1];
    memcpy(<Opaque>result, <Opaque>Ref(value), sizeof<M.Pair>());
    return result;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slot0, <Opaque>slot1];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 100];
    const derived = <M.Derived><Opaque>Ref(data);

    assert(derived.first() == 10, "slot 0");
    const pair = derived.pair(7);
    assert(pair.first == 107 && pair.second == 8, "a record returned by a method of an imported class");
    const exportedPair = M.callPair(derived, 9);
    assert(exportedPair.first == 109 && exportedPair.second == 10, "and by the exporting module");

    print("done.");
}

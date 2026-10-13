import './export_cppvtable_class'

// The "C++ object" made by hand, as in 00cppvtable_class.ts: a table of functions and a tuple
// whose first field points at it.

type ObjectData = [vtable: Opaque, value: s32];

function slot0(self: Opaque): s32 {
    return 10;
}

function slot1(self: Opaque, x: s32): s32 {
    return Deref(<Ref<ObjectData>>self).value + x;
}

function slot2(self: Opaque, x: s32, y: s32): s32 {
    return x * y;
}

function slot3(self: Opaque, x: number): s32 {
    return <s32>(x * 10);
}

function slot4(self: Opaque, x: s32, y: s32): s32 {
    return x - y;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slot0, <Opaque>slot1, <Opaque>slot2, <Opaque>slot3, <Opaque>slot4];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 32];
    const object = <Opaque>Ref(data);

    const derived = <M.Derived>object;
    assert(derived.value == 32, "the first field follows the vtable pointer");
    assert(derived.first() == 10, "slot 0");
    assert(derived.second(5) == 37, "slot 1");
    assert(derived.third(6, 7) == 42, "slot 2");
    assert(M.callSecond(derived, 8) == 40, "the exporting module calls the same slot");
    assert(derived.twice(1) == 66, "a method of TypeScript's own, from the exporting module");
    assert(derived.second(2.5) == 25, "a C++ overload, slot 3");
    assert(derived.second(9, 4) == 5, "the derived class's overload, slot 4");
    assert(M.callSecondDouble(derived, 1.5) == 15, "the exporting module picks the same overload");

    print("done.");
}

// @slot(N): a method of a @cpp class is a C++ virtual method, slot N of the vtable (a base class's
// slots counted), whatever the declaration order; the slots no method names are skipped. A method
// without @slot isn't virtual: with a body it is a method of TypeScript's own (a C++ non-virtual
// method, without a body, is called by its symbol: cppvtable/abi_*.ts). The "C++ object" is made
// by hand, as in 00cppvtable_class.ts.

@cpp
class Base {
    value: s32;
    @slot(3) fourth(x: s32): s32;   // slots 0 and 2 are never called: not declared
    @slot(1) second(): s32;

    // TypeScript's own: not in the vtable
    twice(): s32 {
        return this.second() * 2;
    }

}

@cpp
class Derived extends Base {
    @slot(5) sixth(): s32;          // after the base's slots, slot 4 skipped
}

type ObjectData = [vtable: Opaque, value: s32];

function slotUnused(self: Opaque): s32 {
    return -1;
}

function slot1(self: Opaque): s32 {
    return 10;
}

function slot3(self: Opaque, x: s32): s32 {
    return Deref(<Ref<ObjectData>>self).value * x;
}

function slot5(self: Opaque): s32 {
    return 60;
}

function main() {
    let vtable: Opaque[] = [<Opaque>slotUnused, <Opaque>slot1, <Opaque>slotUnused, <Opaque>slot3, <Opaque>slotUnused, <Opaque>slot5];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 7];
    const object = <Opaque>Ref(data);

    const base = <Base>object;
    assert(base.second() == 10, "slot 1");
    assert(base.fourth(3) == 21, "slot 3, declared first");
    assert(base.twice() == 20, "a method of TypeScript's own calls a slot");

    const derived = <Derived>object;
    assert(derived.sixth() == 60, "slot 5 of a derived class");
    assert(derived.fourth(2) == 14, "a base slot through a derived class");

    print("done.");
}

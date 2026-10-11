// A @cppvtable class is laid over an object made elsewhere (C++): its first word is the vtable, the
// class's methods are that vtable's slots from 0 in declaration order, a base class's first, and
// each is called as fn(this, args). The "C++ object" here is made by hand: a table of functions and
// a tuple whose first field points at it.

@cppvtable
class Base {
    value: s32;          // the object's first field, after the vtable pointer
    first(): s32;        // slot 0
    second(x: s32): s32; // slot 1
}

@cppvtable
class Derived extends Base {
    third(x: s32, y: s32): s32; // slot 2
}

// `abstract` reads the same
@cppvtable
abstract class AbstractBase {
    abstract first(): s32;        // slot 0
    abstract second(x: s32): s32; // slot 1
}

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

function main() {
    let vtable: Opaque[] = [<Opaque>slot0, <Opaque>slot1, <Opaque>slot2];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 32];
    const object = <Opaque>Ref(data);

    const base = <Base>object;
    assert(base.value == 32, "the first field follows the vtable pointer");
    assert(base.first() == 10, "slot 0");
    assert(base.second(5) == 37, "slot 1 gets this");

    const derived = <Derived>object;
    assert(derived.first() == 10, "a base method keeps its slot");
    assert(derived.third(6, 7) == 42, "a derived method follows the base's slots");

    const asBase: Base = derived;
    assert(asBase.second(1) == 33, "a derived object is a base object");

    const abstractBase = <AbstractBase>object;
    assert(abstractBase.first() == 10, "abstract slot 0");
    assert(abstractBase.second(8) == 40, "abstract slot 1");

    print("done.");
}

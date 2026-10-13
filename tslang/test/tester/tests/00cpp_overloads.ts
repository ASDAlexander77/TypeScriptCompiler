// C++ overloads: a @cpp class declares one name several times, each with its own @slot, and a call
// picks one by its arguments, as C++ does at compile time. A derived class overrides one by naming
// its slot, and adds overloads of its own. The "C++ object" is made by hand, as in 00cpp_slot.ts.

@cpp
class Base {
    value: s32;
    @slot(1) write(x: s32): s32;
    @slot(2) write(x: f64): s32;
    @slot(3) write(x: s32, y: s32): s32;
    @slot(4) name(): s32;
}

@cpp
class Derived extends Base {
    @slot(2) write(x: f64): s32;          // overrides the f64 overload only
    @slot(5) write(s: string): s32;       // an overload of its own
}

type ObjectData = [vtable: Opaque, value: s32];

function slot0(self: Opaque): s32 { return -1; }
function slot1(self: Opaque, x: s32): s32 { return 100 + x; }
function slot2(self: Opaque, x: number): s32 { return 200 + <s32>(x * 10); }
function slot3(self: Opaque, x: s32, y: s32): s32 { return 300 + x * y; }
function slot4(self: Opaque): s32 { return Deref(<Ref<ObjectData>>self).value; }
function slot5(self: Opaque, s: string): s32 { return 500 + s.length; }

function main() {
    let vtable: Opaque[] = [<Opaque>slot0, <Opaque>slot1, <Opaque>slot2, <Opaque>slot3, <Opaque>slot4, <Opaque>slot5];
    let data: ObjectData = [<Opaque>Ref(vtable[0]), 7];

    const base = <Base><Opaque>Ref(data);
    const small: u8 = 9;
    const half: number = 0.5;
    assert(base.write(5) == 105, "an integer literal: write(s32)");
    assert(base.write(small) == 109, "a u8: write(s32), an integer for an integer");
    assert(base.write(2.5) == 225, "a fraction: write(f64)");
    assert(base.write(half) == 205, "a number: write(f64)");
    assert(base.write(6, 7) == 342, "two arguments: write(s32, s32)");
    assert(base.name() == 7, "a method that isn't overloaded");

    const derived = <Derived><Opaque>Ref(data);
    assert(derived.write(5) == 105, "a base overload through a derived class");
    assert(derived.write(2.5) == 225, "the overridden overload, in its slot");
    assert(derived.write("abc") == 503, "the derived class's own overload");
    assert(derived.write(2, 3) == 306, "another base overload");

    print("done.");
}

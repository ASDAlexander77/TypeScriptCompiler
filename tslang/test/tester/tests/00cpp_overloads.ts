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

// classes are C++ pointers: a class fits its own type and `T | null` alike, and a derived class
// fits its base as a derived-to-base conversion, below an exact fit
@cpp
class Shape {
    @slot(1) area(): s32;
}

@cpp
class Square extends Shape {
    @slot(2) side(): s32;
}

@cpp
class Canvas {
    @slot(1) draw(x: s32): s32;
    @slot(2) draw(shape: Shape | null): s32;
    @slot(3) fill(shape: Square): s32;
    @slot(4) fill(shape: Shape): s32;
}

// overloads told apart by their number of parameters alone
@cpp
class Mixer {
    @slot(1) mix(a: s32, b: s32): s32;
    @slot(2) mix(a: s32, b: s32, c: s32): s32;
}

type ObjectData = [vtable: Opaque, value: s32];

function slot0(self: Opaque): s32 { return -1; }
function slot1(self: Opaque, x: s32): s32 { return 100 + x; }
function slot2(self: Opaque, x: number): s32 { return 200 + <s32>(x * 10); }
function slot3(self: Opaque, x: s32, y: s32): s32 { return 300 + x * y; }
function slot4(self: Opaque): s32 { return Deref(<Ref<ObjectData>>self).value; }
function slot5(self: Opaque, s: string): s32 { return 500 + s.length; }

function canvasDrawInt(self: Opaque, x: s32): s32 { return 1000 + x; }
function canvasDrawShape(self: Opaque, shape: Opaque): s32 { return 2000; }
function canvasFillSquare(self: Opaque, shape: Opaque): s32 { return 3000; }
function canvasFillShape(self: Opaque, shape: Opaque): s32 { return 4000; }

function mix2(self: Opaque, a: s32, b: s32): s32 { return 10 * a + b; }
function mix3(self: Opaque, a: s32, b: s32, c: s32): s32 { return 100 * a + 10 * b + c; }

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

    let canvasVtable: Opaque[] = [<Opaque>slot0, <Opaque>canvasDrawInt, <Opaque>canvasDrawShape,
        <Opaque>canvasFillSquare, <Opaque>canvasFillShape];
    let canvasData: ObjectData = [<Opaque>Ref(canvasVtable[0]), 0];
    const canvas = <Canvas><Opaque>Ref(canvasData);
    const shape = <Shape><Opaque>Ref(data);
    const square = <Square><Opaque>Ref(data);
    const maybeShape: Shape | null = shape;
    assert(canvas.draw(3) == 1003, "an integer: draw(s32)");
    assert(canvas.draw(maybeShape) == 2000, "a Shape | null: draw(Shape | null)");
    assert(canvas.draw(shape) == 2000, "a Shape: draw(Shape | null)");
    assert(canvas.draw(square) == 2000, "a Square: draw(Shape | null), derived to base");
    assert(canvas.fill(square) == 3000, "a Square: fill(Square) over fill(Shape)");
    assert(canvas.fill(shape) == 4000, "a Shape: fill(Shape)");

    let mixerVtable: Opaque[] = [<Opaque>slot0, <Opaque>mix2, <Opaque>mix3];
    let mixerData: ObjectData = [<Opaque>Ref(mixerVtable[0]), 0];
    const mixer = <Mixer><Opaque>Ref(mixerData);
    assert(mixer.mix(1, 2) == 12, "two arguments: mix(s32, s32)");
    assert(mixer.mix(1, 2, 3) == 123, "three arguments: mix(s32, s32, s32)");

    // `?.`: the call picks among the overloads as a direct call does
    const maybeMixer: Mixer | null = mixer;
    assert(maybeMixer?.mix(4, 5) == 45, "?. with two arguments: mix(s32, s32)");
    assert(maybeMixer?.mix(4, 5, 6) == 456, "?. with three arguments: mix(s32, s32, s32)");
    const maybeDerived: Derived | null = derived;
    assert(maybeDerived?.write(2.5) == 225, "?. on a derived class: its override");
    assert(maybeDerived?.write("abcd") == 504, "?. on a derived class: its own overload");
    assert(maybeDerived?.write(2, 3) == 306, "?. on a derived class: a base overload");
    const noMixer: Mixer | null = null;
    assert(noMixer?.mix(1, 2, 3) === undefined, "?. on null calls nothing");

    print("done.");
}

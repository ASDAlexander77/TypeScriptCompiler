// A @cpp class implemented here (its methods have bodies) as a value (`typeof A`): `new k(n)` through
// it makes an object of the class the value holds, whose vtable is laid out as C++'s, and the value's
// identity is that vtable (a @cpp class has no rtti). A @cpp class made by C++ only can't be a value.

// a C++ interface
@cpp
class Shape {
    @slot(0) area(): s32;
    @slot(1) scaled(k: s32): s32;
}

@cpp
class Square extends Shape {
    constructor(public side: s32) {
        super();
    }

    area(): s32 {
        return this.side * this.side;
    }

    scaled(k: s32): s32 {
        return k * this.area();
    }
}

@cpp
class Triple extends Square {
    area(): s32 {
        return 3 * this.side;
    }
}

@cpp
class Unit extends Shape {
    area(): s32 {
        return 1;
    }

    scaled(k: s32): s32 {
        return k;
    }
}

// as C++ would call it: slot 0 of the object's vtable, the object first
type ObjectHead = [vtable: Ref<Opaque>];

function callSlot0(object: Opaque): s32 {
    const vtable = Deref(<Ref<ObjectHead>>object).vtable;
    const slot0 = <(self: Opaque) => s32>Deref(vtable);
    return slot0(object);
}

function make(k: typeof Square, side: s32): Shape {
    return new k(side);
}

function main() {
    const a = make(Square, 4);
    assert(a.area() == 16 && a.scaled(2) == 32, "made through the value, called through its slots");
    const b = make(Triple, 4);
    assert(b.area() == 12 && b.scaled(2) == 24, "a subclass's slots");
    assert(callSlot0(<Opaque>b) == 12, "its vtable is C++'s");

    const kinds: (typeof Square)[] = [Square, Triple];
    let total: s32 = 0;
    for (const k of kinds) {
        total += new k(2).area();
    }

    assert(total == 4 + 6, "an array of classes");

    const s: typeof Square = Square;
    const t: typeof Square = Triple;
    const u: typeof Square = Square;
    assert(s == u && s != t && t == Triple, "equality: the vtable");

    const one: typeof Shape = Unit;
    assert(new one().scaled(7) == 7, "a C++ interface's value");

    print("done.");
}

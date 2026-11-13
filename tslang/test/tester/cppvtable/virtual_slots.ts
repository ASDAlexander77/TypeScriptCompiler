// One binding for every target: the slots of @virtual methods are numbered by tslang as the
// target's C++ compiler numbers them (abi_host.cpp's VirtualBase, VirtualDerived) - MSVC with one
// destructor slot and overloads grouped in reverse order, the Itanium ABI with two destructor slots
// and the declaration order. Before @virtual, this needed a binding per ABI with its @slot numbers.

@cpp
class VirtualBase {
    @virtual f(): s32;
    @destructor destroy(): void;
    @virtual over(a: s32, b: s32): s32;
    @virtual g(): s32;
    @virtual over(x: f64): s32;
    @virtual over(x: s32): s32;
    @virtual h(): s32;
}

@cpp
class VirtualDerived extends VirtualBase {
    @virtual k(): s32;
    @virtual over(x: f64): s32;     // overrides the base class's, in its slot
    @virtual over(x: s16): s32;     // a new overload
    @virtual m(): s32;
    @virtual k(x: s32): s32;
    @destructor destroy(): void;    // overrides the base class's
    @virtual g(): s32;
}

declare function virtual_derived(): VirtualDerived;
declare function virtual_base(): VirtualBase;
declare function virtual_delete(object: VirtualBase): void;

function main() {
    const base = virtual_base();
    assert(base.f() == 1 && base.g() == 2 && base.h() == 3, "the base class's methods");
    assert(base.over(3, 4) == 107, "over(int, int)");
    const half: f64 = 0.5;
    assert(base.over(half) == 200, "over(double)");
    assert(base.over(<s32>7) == 307, "over(int)");
    virtual_delete(base);

    const derived = virtual_derived();
    assert(derived.f() == 1 && derived.h() == 3, "inherited");
    assert(derived.g() == 20, "an override");
    assert(derived.k() == 4 && derived.k(<s32>6) == 506, "k and its overload, MSVC grouping them");
    assert(derived.m() == 5, "after the overloads");
    assert(derived.over(half) == 201, "an overload overridden");
    assert(derived.over(<s16>9) == 409, "a new overload");
    assert(derived.over(<s32>7) == 307 && derived.over(1, 2) == 103, "the base class's overloads");

    const asBase: VirtualBase = derived;
    assert(asBase.g() == 20 && asBase.over(half) == 201, "through the base class");
    virtual_delete(derived);

    print("done.");
}

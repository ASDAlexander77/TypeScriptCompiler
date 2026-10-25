/// <reference path="fixture_cpp.ts" />

// fixture_cpp.hpp's classes, through the @cpp classes tsbindgen generated for this target's ABI

function main() {
    const square = fx_make_square(3);
    print(square.sides, square.side);       // fields after the vtable pointer, the base's first
    print(square.area());                   // a virtual method, overridden in C++
    square.scale(2);                        // scale(int32_t)
    print(square.area());
    square.scale(0.5);                      // scale(double): an overload, picked by the argument
    print(square.area());

    const shape: Shape = square;
    print(shape.area());                    // through the base class
    const pair = shape.pair(7);             // a trivial record, returned by value
    print(pair.a, pair.b);
    const handle = shape.handle();          // a non-trivial one (@cppnontrivial)
    print(handle.p == <Opaque>shape);
    print(shape.id(5));                     // a non-virtual method, by its C++ declaration (@cppname)
    let extent: Extent = [3, 4];            // fx::Extent, a struct in a namespace: a tuple
    print(shape.fits(Ref(extent), null), shape.fits(Ref(extent), Ref(extent)));   // a reference, a pointer or null

    const small = fx_make_square(2);
    print(square.bigger(small), small.bigger(square));

    // fx::Batch: a SmallVec<Shape *, 3> (its private base's array, then its count), a union as raw
    // words ('A' in the buffer's first byte), a run of bitfields as a raw word (flags 3, mode 2)
    let batch: Batch = [[[square, small, null], 2], [65, 0], [3 + 2 * 8], 1000];
    print(shape.sum(Ref(batch)));
}

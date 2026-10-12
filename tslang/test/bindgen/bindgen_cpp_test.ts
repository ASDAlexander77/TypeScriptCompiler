/// <reference path="fixture_cpp.ts" />

// fixture_cpp.hpp's classes, through the @cpp classes tsbindgen generated for this target's ABI

function main() {
    const square = fx_make_square(3);
    print(square.sides, square.side);       // fields after the vtable pointer, the base's first
    print(square.area());                   // a virtual method, overridden in C++
    square.scale(2);                        // scale(int32_t)
    print(square.area());
    square.scale_2(0.5);                    // scale(double), the second overload
    print(square.area());

    const shape: Shape = square;
    print(shape.area());                    // through the base class
    const pair = shape.pair(7);             // a trivial record, returned by value
    print(pair.a, pair.b);
    const handle = shape.handle();          // a non-trivial one (@cppnontrivial)
    print(handle.p == <Opaque>shape);
    print(shape.id(5));                     // a non-virtual method, by its mangled symbol

    const small = fx_make_square(2);
    print(square.bigger(small), small.bigger(square));
}

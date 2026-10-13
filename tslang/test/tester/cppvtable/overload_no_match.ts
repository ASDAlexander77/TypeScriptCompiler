// A call of a C++ overloaded method takes the overload its arguments fit: one that fits none is an error.
@cpp
class Base {
    @slot(0) write(x: s32): s32;
    @slot(1) write(x: f64): s32;
}

function main(b: Base) {
    b.write("text");
}

// Two overloads the arguments fit as well: an error, as in C++.
@cpp
class Base {
    @slot(0) write(x: s32, y: f64): s32;
    @slot(1) write(x: f64, y: s32): s32;
}

function main(b: Base) {
    b.write(1, 1);
}

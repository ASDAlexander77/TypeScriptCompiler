// An override of a base class's C++ overload names its slot: by name it would take the first's.
@cpp
class Base {
    @slot(0) write(x: s32): s32;
    @slot(1) write(x: f64): s32;
}

@cpp
class Derived extends Base {
    write(x: f64): s32;
}

function main() {
}

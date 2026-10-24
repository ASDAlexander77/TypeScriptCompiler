// A C++ binding is dispatched through the C++ vtable, and its layout is C++'s: it cannot be a value.
@value
@cpp
declare class V {
    m(): int;
}

function main() {
}

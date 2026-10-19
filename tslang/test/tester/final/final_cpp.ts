// A C++ binding is dispatched through the C++ vtable, which @final would bypass.
@final
@cpp
declare class P {
    m(): int;
}

function main() {
}

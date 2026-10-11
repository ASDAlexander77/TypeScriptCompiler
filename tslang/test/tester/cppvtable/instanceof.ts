// `instanceof` asks slot 0, which is a C++ method in a @cppvtable class's vtable.
@cppvtable
class Base {
    first(): s32;
}

class Plain {}

function main() {
    const p = new Plain();
    const b = <Base><Opaque>p;
    print(b instanceof Plain);
}

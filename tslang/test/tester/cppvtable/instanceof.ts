// `instanceof` asks slot 0, which is a C++ method in a @cpp class's vtable.
@cpp
class Base {
    @slot(0) first(): s32;
}

class Plain {}

function main() {
    const p = new Plain();
    const b = <Base><Opaque>p;
    print(b instanceof Plain);
}

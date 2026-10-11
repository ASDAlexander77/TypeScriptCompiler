// An interface over a class object is found through the object's vtable, which is the C++ one.
@cppvtable
class Base {
    first(): s32;
}

interface IFirst {
    first(): s32;
}

function main() {
    const b = <Base><Opaque>0;
    const i: IFirst = b;
    print(i.first());
}

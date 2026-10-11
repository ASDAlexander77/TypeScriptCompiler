// Unboxing an `any` to a @cppvtable class would ask the object what it is.
@cppvtable
class Base {
    first(): s32;
}

function main() {
    const a: any = 1;
    const b = <Base>a;
    print(b.first());
}

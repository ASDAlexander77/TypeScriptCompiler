// An `any` is unboxed by asking the object what it is, which a C++ object can't answer.
@cpp
class Base {
    @slot(0) first(): s32;
}

function main() {
    const b = <Base><Opaque>0;
    const a: any = b;
    print(a);
}

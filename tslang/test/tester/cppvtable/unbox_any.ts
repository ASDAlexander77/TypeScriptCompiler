// Unboxing an `any` to a @cpp class would ask the object what it is.
@cpp
class Base {
    @slot(0) first(): s32;
}

function main() {
    const a: any = 1;
    const b = <Base>a;
    print(b.first());
}

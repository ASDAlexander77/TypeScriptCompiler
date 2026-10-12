// An interface takes a slot in an ordinary class's vtable.
interface IFirst {
    first(): s32;
}

@cpp
class Base implements IFirst {
    @slot(0) first(): s32;
}

function main() {
    print("done.");
}

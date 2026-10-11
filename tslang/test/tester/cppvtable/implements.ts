// An interface takes a slot in an ordinary class's vtable.
interface IFirst {
    first(): s32;
}

@cppvtable
class Base implements IFirst {
    first(): s32;
}

function main() {
    print("done.");
}

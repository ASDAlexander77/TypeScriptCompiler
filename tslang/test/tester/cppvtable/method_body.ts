// A method of a @cppvtable class is a slot of the C++ vtable, not code of its own.
@cppvtable
class Base {
    first(): s32 { return 1; }
}

function main() {
    print("done.");
}

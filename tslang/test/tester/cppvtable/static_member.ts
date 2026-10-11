// A static member would take a vtable slot.
@cppvtable
class Base {
    static count: s32;
    first(): s32;
}

function main() {
    print("done.");
}

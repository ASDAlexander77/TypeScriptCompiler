// A static member would take a vtable slot.
@cpp
class Base {
    static count: s32;
    @slot(0) first(): s32;
}

function main() {
    print("done.");
}

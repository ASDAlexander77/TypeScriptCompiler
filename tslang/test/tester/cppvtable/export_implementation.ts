// A class implementing C++ methods isn't exported: an importer would get its declaration only, with
// the methods but not the vtable they are in.
@cppvtable
export class Base {
    first(): s32;
}

@cppvtable
export class Implementation extends Base {
    first(): s32 { return 1; }
}

function main() {
    print("done.");
}

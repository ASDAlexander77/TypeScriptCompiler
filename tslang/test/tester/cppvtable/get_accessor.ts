// An accessor is a generated method, and its slot wouldn't follow the declaration order.
@cppvtable
class Base {
    get size(): s32;
    first(): s32;
}

function main() {
    print("done.");
}

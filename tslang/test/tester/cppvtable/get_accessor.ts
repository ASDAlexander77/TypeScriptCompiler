// An accessor is a generated method, and its slot wouldn't follow the declaration order.
@cpp
class Base {
    get size(): s32;
    @slot(0) first(): s32;
}

function main() {
    print("done.");
}

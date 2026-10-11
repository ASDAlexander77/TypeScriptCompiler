// A field initializer is code run by a constructor, which a @cppvtable class doesn't have.
@cppvtable
class Base {
    value: s32 = 1;
    first(): s32;
}

function main() {
    print("done.");
}

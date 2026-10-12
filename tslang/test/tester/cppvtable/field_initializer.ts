// A field initializer is code run by a constructor, which a @cpp class doesn't have.
@cpp
class Base {
    value: s32 = 1;
    @slot(0) first(): s32;
}

function main() {
    print("done.");
}

// Compiled under rc, own and none: a @cpp reference is a C++ pointer, which no model counts, so
// nothing is written in the word before the C++ object.
@cpp
class Base {
    @slot(0) first(): s32;
}

function call(b: Base) {
    print(b.first());
}

function main() {
    print("done.");
}

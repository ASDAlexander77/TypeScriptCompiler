// Compiled with -mm=rc: retains and releases would write the count in the word before a C++
// object. The same file compiles with -mm=none.
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

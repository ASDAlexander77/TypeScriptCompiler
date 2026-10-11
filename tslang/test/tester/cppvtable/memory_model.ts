// Compiled with -mm=rc: retains and releases would write the count in the word before a C++
// object. The same file compiles with -mm=none.
@cppvtable
class Base {
    first(): s32;
}

function call(b: Base) {
    print(b.first());
}

function main() {
    print("done.");
}

// A method with a body implements a C++ method (the class is called by C++), and can't return a
// struct yet: its body would have to write it through the hidden pointer.
type Pair = [first: s32, second: s32];

@cppvtable
class Base {
    pair(x: s32): Pair;
}

@cppvtable
class Implementation extends Base {
    pair(x: s32): Pair { return [x, x]; }
}

function main() {
    print("done.");
}

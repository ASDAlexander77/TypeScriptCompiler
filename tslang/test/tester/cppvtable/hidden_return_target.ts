// Compiled for Linux: the Itanium ABI returns a record from a method otherwise than MSVC, through
// a pointer before `this`, and only some records.
type Pair = [first: s32, second: s32];

@cppvtable
class Methods {
    pair(x: s32): Pair;
}

function main() {
    print("done.");
}

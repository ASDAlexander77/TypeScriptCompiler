// Compiled for AArch64 Linux: the Itanium ABI passes the hidden pointer a record is returned
// through in X8 there, which an ordinary argument isn't.
type Pair = [first: s32, second: s32];

@cppvtable
class Methods {
    pair(x: s32): Pair;
}

function main() {
    print("done.");
}

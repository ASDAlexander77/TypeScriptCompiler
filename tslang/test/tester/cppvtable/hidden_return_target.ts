// Compiled for 32-bit x86 Linux: the Itanium ABI passes the hidden pointer a record is returned
// through on the stack there, and the callee pops it, which isn't done.
type Pair = [first: s32, second: s32];

@cppvtable
class Methods {
    pair(x: s32): Pair;
}

function main() {
    print("done.");
}

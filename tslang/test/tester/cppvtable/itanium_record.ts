// Compiled for x86-64 Linux: a trivial record of two 4-byte fields comes back packed into RAX,
// which a tuple of them isn't; without @cppnontrivial it is an error, not a wrong call.
type Pair = [first: s32, second: s32];

@cppvtable
class Methods {
    pair(x: s32): Pair;
}

function main() {
    print("done.");
}

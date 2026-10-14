// Compiled for x86-64 Linux: a trivial record of two floats comes back in XMM0, which isn't done;
// without @cppnontrivial it is an error, not a wrong call.
type Pair = [first: f32, second: f32];

@cpp
class Methods {
    @slot(0) pair(x: s32): Pair;
}

function main() {
    print("done.");
}

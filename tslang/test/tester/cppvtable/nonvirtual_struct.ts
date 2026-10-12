// A C++ non-virtual method returning a record isn't supported yet: its symbol would have to be
// declared with the hidden result pointer.
type Pair = [a: Opaque | null, b: Opaque | null];

@cpp
class Base {
    @linkname("?pair@Base@@QEAA?AUPair@@XZ")
    pair(): Pair;
}

function main() {
}

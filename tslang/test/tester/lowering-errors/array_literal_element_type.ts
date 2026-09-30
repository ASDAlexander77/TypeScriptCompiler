// Lowering cannot read the array out of a number[] | null, and says so with an error. That error has
// to stop the compile: before it did, the object file was written and the program run.
// (The array-to-string cast this test used first hits llvm_unreachable in a Debug build before it
// reports anything. Its second, a const array literal stored as a number[] | null, compiles now.)
function lengthOf(p: number[] | null) {
    return p ? p.length : -1;
}

function main() {
    print(lengthOf(null));
    print("done.");
}

// Lowering cannot build this array literal as a number[] | null, and says so with an error. That
// error has to stop the compile: before it did, the object file was written and the program run.
// (The array-to-string cast this test used first hits llvm_unreachable in a Debug build before it
// reports anything; this error is reported the same way in both.)
function main() {
    let a: number[] | null = [1, 2];
    print("done.");
}

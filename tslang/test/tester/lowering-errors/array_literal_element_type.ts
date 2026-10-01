// Lowering cannot widen a number[] to an any[] (it would need a per-element conversion), and says so
// with an error. That error has to stop the compile: before it did, the object file was written and
// the program run.
// (The array-to-string cast this test used first hits llvm_unreachable in a Debug build before it
// reports anything. Its next two, a const array literal stored as a number[] | null and the array
// read out of one, compile now.)
function count(p: any[]) {
    return p.length;
}

function main() {
    let a: number[] = [1, 2];
    print(count(a));
    print("done.");
}

// Lowering cannot widen a number[] to a (number | string)[] (it would need a per-element conversion), and
// says so with an error. That error has to stop the compile: before it did, the object file was written
// and the program run.
// (The array-to-string cast this test used first hits llvm_unreachable in a Debug build before it
// reports anything. Its next two, a const array literal stored as a number[] | null and the array
// read out of one, compile now, and so does its third, a number[] given as an any[]: a copy, each
// element boxed.)
function count(p: (number | string)[]) {
    return p.length;
}

function main() {
    let a: number[] = [1, 2];
    print(count(a));
    print("done.");
}

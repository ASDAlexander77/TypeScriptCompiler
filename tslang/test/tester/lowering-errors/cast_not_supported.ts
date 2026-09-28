// Lowering has no cast from an array to a string, and says so with an error. That error has to
// stop the compile: before it did, the program was emitted and run anyway and crashed.
function main() {
    let a: number[] | null = [1, 2];
    print("a=" + a);
}

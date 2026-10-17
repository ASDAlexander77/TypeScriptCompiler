// A for...of over something that isn't iterable was reported at the start of the file: the loop
// the compiler makes for it had no position.

function main() {
    const n = 5;
    for (const x of n) {
        print(x);
    }
}

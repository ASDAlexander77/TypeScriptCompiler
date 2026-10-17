// A for...in over something without a length was reported at the start of the file: the loop
// the compiler makes for it had no position.

function main() {
    const n = 5;
    for (const k in n) {
        print(k);
    }
}

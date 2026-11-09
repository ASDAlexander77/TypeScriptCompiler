// A symbol is a compile-time name so far: using one at run time is an error that says so.
function main() {
    const k = Symbol.iterator;
    print(k);
}

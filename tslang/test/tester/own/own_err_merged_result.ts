// -mm=own rejects: the result of a `?:` is either branch's value, and nothing says which one
// owns it.
function max(a: string, b: string) {
    return a > b ? a : b;
}
function main() {
    print(max("a" + 1, "b" + 2));
}

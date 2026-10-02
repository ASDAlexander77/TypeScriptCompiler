// -mm=own rejects: `identity` returns its parameter, a result its callers would have to know
// borrows the argument, but an exported function can be called from another module, which does not.
export function identity(x: string) {
    return x;
}
function main() {
    print(identity("x"));
}

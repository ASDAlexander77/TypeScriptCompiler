// The built-in `Symbol` read as a value is an error at its place: it is a name `Symbol(...)`, `Symbol.for(...)` and
// the well-known keys are resolved through, with nothing behind it ("'llvm.mlir.addressof' op must reference a global").
function main() {
    const x: any = Symbol("a");
    if (x.constructor === Symbol) print("symbol");
}

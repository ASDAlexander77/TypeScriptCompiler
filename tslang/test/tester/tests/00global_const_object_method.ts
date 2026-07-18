// regression test: a top-level (global) `const` object literal with a method that
// mutates `this` used to crash the compiler (0xC0000005) at JIT materialization time.
// Root cause: GlobalOpLowering's visitorAllOps walk decided whether a global needs a
// runtime constructor function or can be a plain static LLVM `constant` initializer,
// but had no case for a ConstantOp whose TupleType/ConstTupleType result has a
// bound-method field -- such a field lowers via LLVM::AddressOfOp + InsertValueOp,
// which is not valid inside a static global initializer region, only inside real code.
// `let` at global scope, and `const` declared locally inside a function, were both
// unaffected (they already go through -- or don't need -- the constructor path).
const obj = {
    count: 0,
    inc() { this.count = this.count + 1; },
    greet() { print("hi"); }
};

function main() {
    obj.greet();
    obj.inc();
    obj.inc();
    assert(obj.count == 2);
    print("done.");
}

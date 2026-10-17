// A rest parameter of a number type reached the MLIR verifier, which reported the array made
// of the arguments at the call and the module at 1:1.

function sum(...a: int[]) {}

function f(...a: int) {}

function main() {
    sum(1, 2);
    f(1, 2);
}

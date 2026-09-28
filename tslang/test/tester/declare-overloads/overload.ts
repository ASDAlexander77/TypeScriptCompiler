// A declared function is one external symbol with one signature: a second signature is an error
// that says so. It used to be "redefinition of symbol" from the verifier.
declare function abs(x: i32): i32;
declare function abs(x: i32, y: i32): i32;

function main() {
    print(abs(-3));
}

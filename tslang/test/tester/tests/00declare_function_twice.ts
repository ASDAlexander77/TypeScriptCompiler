// The same declared function twice - one C function declared in two .d.ts files - is one
// declaration. It used to be "redefinition of symbol".
declare function abs(x: i32): i32;
declare function abs(x: i32): i32;

function main() {
    assert(abs(-3) == 3);
    print("done.");
}

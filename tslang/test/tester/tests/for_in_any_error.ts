// TSLANG_ISSUES 88: an `any` has no keys known at compile time for `for...in` to iterate (it crashed, or
// failed with "Can't resolve property 'length' of type any")
var a: any;
var expr: any;

function main() {
    for (a in expr) {
    }

    print("done.");
}

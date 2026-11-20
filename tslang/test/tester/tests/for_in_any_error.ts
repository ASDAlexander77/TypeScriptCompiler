// TSLANG_ISSUES 88: an `any` has no keys known at compile time for `for...in` to iterate (it crashed, or
// failed with "Can't resolve property 'length' of type any"). Its keys are read at run time now
// (00for_in_any.ts), but not under -mm=own, where this is compiled to say so.
var a: any;
var expr: any;

function main() {
    for (a in expr) {
    }

    print("done.");
}

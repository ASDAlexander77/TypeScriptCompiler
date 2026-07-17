// regression test for #238: getFieldTypeByFieldName used to hit llvm_unreachable
// when the `in` operator (or structural type checks that reuse the same lookup)
// was asked about a name that is a method/extension-function name rather than a
// real data field -- e.g. "push" on an array, "fromCharCode"/"charAt" on a string,
// or a method name on a class instance. All five branches (Array, ConstArray,
// String, Interface, Class) must return "not found" instead of aborting.

class Point {
    x: number;
    y: number;

    constructor(x: number, y: number) {
        this.x = x;
        this.y = y;
    }

    dist(): number {
        return this.x * this.x + this.y * this.y;
    }
}

function main() {
    const trees = ["redwood", "bay", "cedar"];

    // real data fields/indices are found
    assert("length" in trees);
    assert(0 in trees);

    // method/extension names are not data fields, must resolve to false, not abort
    assert(!("push" in trees));
    assert(!("pop" in trees));
    assert(!("entries" in trees));
    assert(!("madeUpName" in trees));

    const s = "hello";

    assert("length" in s);
    assert(!("charAt" in s));
    assert(!("fromCharCode" in s));

    const p = new Point(3, 4);

    assert("x" in p);
    assert("y" in p);
    assert("dist" in p);
    assert(!("madeUpField" in p));

    print("done.");
}

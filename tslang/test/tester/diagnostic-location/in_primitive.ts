// The right-hand side of `in` that is a number had no fields to look in, and the test was
// quietly false.

function main() {
    const s = "hello";
    print("length" in s);
    print("a" in 5);
}

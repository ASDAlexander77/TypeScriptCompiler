// A member after a string member, with no initializer, had no value to count on from: it
// quietly took the count of the members before.

enum E {
    N,
    A = "x",
    B,
}

function main() {
    print(E.B);
}

// an `any` cast to an object type: the check made for the cast names the type in a string, and the quotes of a
// literal-typed field ended that string (the check did not compile)
function main() {
    const a: any = { k: "x" };
    const v = a as { k: "x" };
    assert(v.k == "x", "literal field read back");

    print("done.");
}

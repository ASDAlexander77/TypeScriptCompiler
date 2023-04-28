function main() {
    eqG();
    print("done.");
}

function eq<A, B>(a: A, b: B) { return a == b as any as A }

function eqG() {
    assert(eq("2", 2), "2")
    assert(eq(2, "2"), "2'")
    //assert(!eq("null", null), "=1") // TODO: null ref here
    assert(!eq(null, "null"), "=2")
    assert(!eq("2", 3), "=3")
}

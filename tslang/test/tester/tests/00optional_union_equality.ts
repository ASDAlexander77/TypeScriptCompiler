// A union that may be undefined (`number | string | undefined`, an optional of the union) compared with a value:
// equal when it holds a member equal to the value, never when it is empty. The lowering met the union inside the
// optional and crashed the compiler.
function k(i: number): number | string | undefined {
    return i == 0 ? "a" : i == 1 ? 5 : undefined;
}

function main() {
    assert(k(0) == "a");
    assert(!(k(1) == "a"));
    assert(!(k(2) == "a"));
    assert(k(1) == 5);
    assert(k(0) != 5);
    assert(k(2) != 5);
    assert(k(0) === "a");
    assert(k(2) !== "a");
    assert("a" == k(0));
    print("done.");
}

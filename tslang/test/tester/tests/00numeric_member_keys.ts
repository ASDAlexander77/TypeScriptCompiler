// `C[0]` of a member named `0` (`static 0 = 1`, `1 = "one"`): the member, as `C["0"]` is - read as an indexer's
// key it was "can't resolve property/field/base '.index'".
class C {
    static "foo" = 3;
    static 0 = 1;
    1 = "one";

    x = C[0];
}

function main() {
    assert(C["foo"] == 3);
    assert(C["0"] == 1 && C[0] == 1);
    const c = new C();
    assert(c[1] == "one" && c["1"] == "one");
    assert(c.x == 1);
    print("done.");
}

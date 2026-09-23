// A nested array literal's inner arrays are mutable. Turning the literal into an array copied only
// the outer data to the heap: the inner arrays kept pointing at the literal's constant data, so a
// push onto one reallocated a global (a crash in the allocator) and an element write wrote one.
const moduleNested = [[1, 2], [3]];

function main() {
    const nested = [[1, 2], [3]];
    nested[1].push(4);
    nested[0][0] = 9;
    assert(nested[1].length == 2, "const: push onto an inner array");
    assert(nested[1][1] == 4, "const: pushed element");
    assert(nested[0][0] == 9, "const: element write in an inner array");

    let reassignable = [[5], [6, 7]];
    reassignable[0].push(8);
    assert(reassignable[0].length == 2, "let: push onto an inner array");
    assert(reassignable[0][1] == 8, "let: pushed element");

    const deep = [[[1]], [[2, 3]]];
    deep[1][0].push(4);
    assert(deep[1][0].length == 3, "three levels: push onto the innermost array");

    moduleNested[0].push(10);
    assert(moduleNested[0].length == 3, "module: push onto an inner array");
    assert(moduleNested[0][2] == 10, "module: pushed element");

    // a fresh literal each time: the first call's writes are not seen by the second
    for (let i = 0; i < 2; i++) {
        const fresh = [[0]];
        assert(fresh[0][0] == 0, "fresh literal per iteration");
        fresh[0][0] = 1;
    }

    print("done.");
}

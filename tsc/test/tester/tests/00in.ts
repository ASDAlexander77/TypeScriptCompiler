function main() {
    // Arrays
    const trees = ["redwood", "bay", "cedar", "oak", "maple"];

    assert(!(-1 in trees));
    assert(0 in trees);
    assert(3 in trees);
    assert(!(5 in trees));
    assert(!(6 in trees));

    for (let i = 0; i in trees; i++) {
        print(trees[i]);
    }

    const obj = { test: 1 };

    assert("test" in obj);
    assert(!("test1" in obj));

    print("done.");
}

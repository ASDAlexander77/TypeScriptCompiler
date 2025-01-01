function main() {

    let count = 0;

    [...[1, 2, "Hello"]].forEach(
        x => {
            if (typeof x == "string") { print(x); count++; }
            if (typeof x == "s32") { print(x); count++; }
        }
    );

    assert(count == 3);

    const v = [...([1, 2, 3, 4].filter(x => x % 2))];
    print( v.length );

    // it is array result is correct, if it is tuple result will not be correct
    assert (v.length == 2);

    print("done.");
}
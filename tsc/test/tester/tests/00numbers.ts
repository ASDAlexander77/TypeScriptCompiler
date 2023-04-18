function main() {
    const previouslyMaxSafeInteger = 9007199254740991n;

    print(previouslyMaxSafeInteger, typeof previouslyMaxSafeInteger);

    const b1 = 0b11111111111111111111111111111111111111111111111111111;
    print(b1, typeof b1);

    assert(b1 === previouslyMaxSafeInteger);

    const b2 = 0o377777777777777777;
    print(b2, typeof b2);

    assert(b2 === previouslyMaxSafeInteger);

    const b3 = 0x1fffffffffffff;
    print(b3, typeof b3);

    assert(b3 === previouslyMaxSafeInteger);

    const b4 = 9007199254740991;
    print(b4, typeof b4);

    assert(b4 === previouslyMaxSafeInteger);

    const b5 = 0b11111111111111111111111111111111111111111111111111111n;
    print(b5, typeof b5);

    assert(b5 === previouslyMaxSafeInteger);

    const b6 = 0x1fffffffffffffn;
    print(b6, typeof b6);

    assert(b6 === previouslyMaxSafeInteger);

    const b7 = 0x1fffffffffffff;
    print(b7, typeof b7);

    assert(b7 === previouslyMaxSafeInteger);

    const b8 = 0o377777777777777777n;
    print(b8, typeof b8);

    assert(b8 === previouslyMaxSafeInteger);

    print("done.");
}

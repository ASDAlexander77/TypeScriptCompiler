function main1() {
    for await (const v of [1, 2, 3, 4, 5]) {
        // TODO: if I add count++, it will work
        print(v);
    }
}

function main2() {
    let count = 0;
    for await (const v of [1, 2, 3, 4, 5]) {
        print(v);
        assert(v >= 1 && v <= 5);
        count++;
    }

    assert(count == 5);
}

function main() {
    main1();
    main2();

    print("done.");
}

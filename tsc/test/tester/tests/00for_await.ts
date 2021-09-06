function main() {
    let count = 0;
    for await (const v of [1, 2, 3, 4, 5]) {
        print(v);
        assert(v >= 1 && v <= 5);
        count++;
    }

    assert(count == 5);

    print("done.");
}

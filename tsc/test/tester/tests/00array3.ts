function main() {
    const trees = [[1], [2, 3], [4, 5, 6]];

    for (const a of trees) {
        print("array");
        for (const b of a) {
            print(b);
        }
    }

    print("done.");
}

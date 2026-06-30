function main() {
    const trees = [
        [1, "redwood", [1]],
        [2, "bay", [1, 2]],
        [3, "cedar", [1, 2, 3]],
        [4, "oak", [1, 2, 3, 4]],
        [5, "maple", [1, 2, 3, 4, 5]],
    ];

    for (const [k, v, a] of trees) {
        print(k, v);

        for (const ai of a) {
            print(ai);
        }
    }

    print("done.");
}

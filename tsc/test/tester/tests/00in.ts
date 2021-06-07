function main() {
    // Arrays
    const trees = ["redwood", "bay", "cedar", "oak", "maple"];
    print(0 in trees); // returns true
    print(3 in trees); // returns true
    print(6 in trees); // returns false

    for (let i = 0; i in trees; i++) {
        print(trees[i]);
    }

    print("done.");
}

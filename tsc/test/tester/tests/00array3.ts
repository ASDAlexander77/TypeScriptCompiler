function main() {
    // example 1
    const trees = [[1], [2, 3], [4, 5, 6]];

    for (const a of trees) {
        print("array");
        for (const b of a) {
            print(b);
        }
    }

    // example 2
    const x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const a = [x, x, x, x, x];

    for (let i = 0, j = 9; i <= j; i++, j--)
        //                                ^
        print("a[" + i + "][" + j + "]= " + a[i][j]);

    print("done.");
}

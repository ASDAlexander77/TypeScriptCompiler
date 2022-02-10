function main() {
    let arr = [1, 2, 3];

    for (const v of arr.map(x => x + 1)) print(v);

    print("done.");
}
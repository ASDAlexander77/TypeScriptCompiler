function main() {
    let arr = ["asd", "asd2"];

    for (const e of arr.map((e) => e + "_").map((e2) => e2 + "2")) print(e);

    print("done.");
}
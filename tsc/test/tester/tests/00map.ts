function main() {
    let arr = [1, 2, 3];

    for (const v of arr.map(x => x + 1)) print(v);

    let arrS = ["asd", "asd2"];
    for (const e of arrS.map((e) => e + "_")) print(e);

    print("done.");
}
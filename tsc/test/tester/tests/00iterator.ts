function main() {
    let arr = [1, 2, 3];

    const it = (function* iter() { for (const v of arr) yield ((x: typeof v) => x + 1)(v); })();

    for (const v of it) print(v);

    print("done.");
}
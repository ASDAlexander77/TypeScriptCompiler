function main() {
    let arr = [1, 2, 3];

    const it = (function* () { for (const v of arr) yield ((x: typeof v) => x + 1)(v); })();

    for (const v of it) print(v);

    const it2 = (function* () { for (const v of arr) yield (<T>(x: T) => x + 1)(v); })();

    for (const v of it2) print(v);

    print("done.");
}
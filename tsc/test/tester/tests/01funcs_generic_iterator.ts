function* Map<T, R>(thisArg: T[], f: (i: T) => R) {
    for (const v of thisArg) yield f(v);
}

function main() {
    let count = 0;
    for (const v of [1, 2, 3].Map((i) => { count++; return i + 1; })) {
        print(v);
    }

    assert(count == 3);
    print("done.");
}
function* map<T, R>(a: T[], f: (i: T) => R) {
    for (const v of a) yield f(v);
}

function main() {
    let count = 0;
    for (const v of map([1, 2, 3], (i) => { count++; return i + 1; })) {
        print(v);
    }

    assert(count == 3);
    print("done.");
}
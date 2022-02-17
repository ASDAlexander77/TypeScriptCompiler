interface Iterable<T> {
    next: () => { value: T, done: boolean }
};

function* g(): Iterable<string> {
    for (let i = 0; i < 100; i++) {
        yield ""; // string is assignable to string
    }
}

function main() {
    for (const v of g()) {
        print(v);
    }

    print("done.");
}
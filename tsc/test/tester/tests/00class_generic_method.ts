class Lib {
    max<T>(a: T, b: T): T {
        return a > b ? a : b;
    }
}

function main() {
    const c = new Lib();
    assert(c.max(10, 20) == 20);
    assert(c.max("a", "b") == "b");
    assert(c.max<number>(10, 20) == 20);

    print("done.");
}
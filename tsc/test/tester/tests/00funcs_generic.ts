function test<T>(t: T) {
    return t;
}

function main() {
    assert(test<number>(10) == 10);
    assert(test<string>("Hello") == "Hello");
    print("done.");
}
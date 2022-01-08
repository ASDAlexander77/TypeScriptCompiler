function test<T>(t: T) {
    return t;
}

function main() {
    print(test<number>(11), test<string>("Hello1"));

    assert(test<number>(10) == 10);
    assert(test<string>("Hello") == "Hello");
    print("done.");
}
function test<T>(t: T) {
    return t;
}

function fib<T>(n: T) {
    return n <= 2 ? n : fib(n - 1) + fib(n - 2);
}

function main() {
    print(test<number>(11), test<string>("Hello1"));

    assert(test<number>(10) == 10);
    assert(test<string>("Hello") == "Hello");

    assert(fib(5) == 8);

    print("done.");
}
import './export_function'

function main() {
    assert(M.add(2, 3) == 5);

    assert(M.greet("World") == "Hello, World!");
    assert(M.greet("World", "Hi") == "Hi, World!");

    assert(M.maybeDouble(5) == 10);
    assert(M.maybeDouble(5, 3) == 15);

    M.noop();

    assert(M.addTwice(2, 3) == 10);

    print("done.");
}

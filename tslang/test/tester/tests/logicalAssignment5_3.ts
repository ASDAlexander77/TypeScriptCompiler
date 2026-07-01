function foo1(f?: (a: number) => number) {
    f ??= ((a) => a)
    assert(f(42) == 42);
}

function foo2(f?: (a: number) => number) {
    f ||= ((a) => a)
    assert(f(42) == 42);
}

function foo3(f?: (a: number) => number) {
    f &&= ((a) => a)
    assert(f(42) == 42);
}

function bar1(f?: (a: number) => number) {
    f ??= (f, ((a) => a))
    assert(f(42) == 42);
}

function bar2(f?: (a: number) => number) {
    f ||= (f, ((a) => a))
    assert(f(42) == 42);
}

function bar3(f?: (a: number) => number) {
    f &&= (f, ((a) => a))
    assert(f(42) == 42);
}

function main() {

    foo1();
    foo2();
    foo3((a) => a * 2);

    bar1();
    bar2();
    bar3((a) => a * 2);

    print("done.");
}
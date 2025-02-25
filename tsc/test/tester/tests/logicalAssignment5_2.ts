function foo1 (f?: (a: number) => number) {
    f ??= ((a: number) => (assert(a == 42), a))
    f(42)
}

function foo2 (f?: (a: number) => number) {
    f ||= ((a: number) =>  (assert(a == 42), a))
    f(42)
}

function foo3 (f?: (a: number) => number) {
    f &&= ((a: number) =>  (assert(a == 42), a))
    f(42)
}

function bar1 (f?: (a: number) => number) {
    f ??= (f, (a =>  (assert(a == 42), a)))
    f(42)
}

function bar2 (f?: (a: number) => number) {
    f ||= (f, (a =>  (assert(a == 42), a)))
    f(42)
}

function bar3 (f?: (a: number) => number) {
    f &&= (f, (a =>  (assert(a == 42), a)))
    f(42)
}

function main() {
    foo1();
    foo2();
    foo3();

    bar1();
    bar2();
    bar3();

    print("done.");
}
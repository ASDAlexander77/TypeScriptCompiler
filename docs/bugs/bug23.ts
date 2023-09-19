function foo1 (f?: (a: number) => number) {
    f ??= ((a: number) => a)
    f(42)
}

function foo2 (f?: (a: number) => number) {
    f ||= ((a: number) => a)
    f(42)
}

function foo3 (f?: (a: number) => number) {
    f &&= ((a: number) => a)
    f(42)
}

function bar1 (f?: (a: number) => void) {
    f ??= (f, (a => a))
    f(42)
}

function bar2 (f?: (a: number) => void) {
    f ||= (f, (a => a))
    f(42)
}

function bar3 (f?: (a: number) => void) {
    f &&= (f, (a => a))
    f(42)
}

function main() {
    print("done.");
}
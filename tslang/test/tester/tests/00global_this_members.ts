// `globalThis.x` where no `globalThis` is declared: the global `x` ("can't resolve name: globalThis"), as a value and
// in a type query.
let counter = 5;

function twice(n: number) {
    return n * 2;
}

const g: typeof globalThis.counter = 3;

function main() {
    assert(globalThis.counter == 5);
    assert(globalThis.twice(4) == 8);
    assert(g == 3);
    print("done.");
}

// TSLANG_ISSUES 37, 38: a function named Symbol crashed the compiler. It is an ordinary function now, and
// the well-known names keep working beside it.

function Symbol(description: string): int {
    return description.length;
}

class Numbers {
    *[Symbol.iterator]() {
        yield 7;
        yield 8;
    }
}

function main() {
    let sum = 0;
    for (const v of new Numbers()) {
        sum += v;
    }

    assert(sum == 15);
    assert(Symbol("four") == 4);
    assert(typeof Symbol.iterator == "symbol");

    print("done.");
}

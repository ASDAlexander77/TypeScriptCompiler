// Falling off the end of a function returns undefined, so a function that can do so beside value returns infers
// `T | undefined`; one whose every path returns or throws keeps `T`
function ifWithoutElse(c: boolean) {
    if (c) return "a";
}

function ifElse(c: boolean) {
    if (c) return "a";
    else return "b";
}

function switchWithDefault(n: number) {
    switch (n) {
        case 1: return "one";
        default: return "many";
    }
}

function switchWithoutDefault(n: number) {
    switch (n) {
        case 1: return "one";
        case 2: return "two";
    }
}

function infiniteLoop(n: number) {
    while (true) {
        if (n > 0) return n;
        n++;
    }
}

function loopWithBreak(n: number) {
    for (;;) {
        if (n > 2.5) break;
        n++;
    }

    return n;
}

function tryCatch(c: boolean) {
    try {
        if (c) return 1;
        throw 1;
    } catch {
        return 2;
    }
}

function returnOrThrow(c: boolean) {
    if (c) {
        return "x";
    }

    throw "e";
}

function loopReturn(n: number) {
    for (const k of [1, 2]) {
        if (k == n) return k;
    }
}

function declared(n: number): number | undefined {
    if (n > 0) return n;
}

function main() {
    assert(ifWithoutElse(false) === undefined, "if without else: falls off");
    assert(ifWithoutElse(true) == "a", "if without else: value");
    assert(ifElse(false) == "b", "if/else");
    assert(switchWithDefault(5) == "many", "switch with default");
    assert(switchWithoutDefault(3) === undefined, "switch without default: falls off");
    assert(switchWithoutDefault(2) == "two", "switch without default: value");
    assert(infiniteLoop(1) == 1, "while (true)");
    assert(loopWithBreak(0) == 3, "for (;;) with break");
    assert(tryCatch(false) == 2, "try/catch");
    assert(returnOrThrow(true) == "x", "return or throw");
    assert(loopReturn(5) === undefined, "loop: falls off");
    assert(loopReturn(2) == 2, "loop: value");
    assert(declared(-1) === undefined, "declared T | undefined: falls off");
    assert(declared(2) == 2, "declared: value");

    // these keep T: a value of a type that cannot be undefined
    const s: string = ifElse(true) + switchWithDefault(1) + returnOrThrow(true);
    assert(s == "aonex", "always-returning functions keep T");

    print("done.");
}

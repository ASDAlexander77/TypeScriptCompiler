// A `?:`, `&&` or `while` whose condition folds at compile time (a `typeof` of a value whose type
// is known) does not generate the branch that never runs. That branch narrows the tested value, so
// for an array narrowed to `string` it cast the array to a string and crashed the compiler.
function conditional<T>(x: T) {
    return typeof x === "string" ? x.length : -1;
}

function conditionalElse<T>(x: T) {
    return typeof x !== "string" ? -1 : x.length;
}

function and<T>(x: T) {
    return typeof x === "string" && x.length > 1;
}

function loop<T>(x: T) {
    let n = 0;
    while (typeof x === "string" && n < x.length) {
        n++;
    }

    return n;
}

function main() {
    assert(conditional("abc") == 3, "?: string");
    assert(conditional([1, 2]) == -1, "?: array");
    assert(conditionalElse("abcd") == 4, "?: else string");
    assert(conditionalElse(<string[]>["a"]) == -1, "?: else array");

    assert(and("abc"), "&& string");
    assert(!and([1, 2]), "&& array");

    assert(loop("ab") == 2, "while string");
    assert(loop([1, 2, 3]) == 0, "while array");

    // the branch that is not generated still gives the expression its type
    const text = "abc";
    let v = typeof text === "string" ? 1 : "one";
    v = "one";
    assert(v == "one", "?: keeps the type of both branches");

    print("done.");
}

// `typeof x === "<name>"` on a value whose type is known at compile time is folded to a
// constant, so a branch that can never run is not generated. Before the fold, the branch
// below that prints an array as a string reached LLVM lowering and crashed the compiler.
function describe<T>(x: T) {
    if (typeof x === "string") {
        print("string: ", x);
        return 1;
    }

    if ("array" === typeof x) {
        return 2;
    }

    if (typeof x !== "boolean") {
        return 3;
    }

    return 4;
}

function main() {
    assert(describe("abc") == 1, "string");
    assert(describe([]) == 2, "array");
    assert(describe(<string[]>["a", "b"]) == 2, "string array");
    assert(describe(2.5) == 3, "number");
    assert(describe(true) == 4, "boolean");

    // `any` is only known at run time and stays a run-time check
    let a: any = "text";
    assert(typeof a === "string", "any holding a string");
    a = 2.5;
    assert(typeof a !== "string", "any holding a number");

    print("done.");
}

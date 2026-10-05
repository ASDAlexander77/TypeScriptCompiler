// an empty string is falsy (#509): a string tested as a boolean was a non-null pointer test, so "" was true
// in `if`, `while`, `!`, `?:`, `&&` and `||`, plain, optional, nullable and as `any`
let globalEmpty = "";
const globalPick = "" ? 1 : 2;

function truthy(s: string) {
    return s ? true : false;
}

function truthyIf(s: string) {
    if (s) {
        return true;
    }

    return false;
}

function truthyOptional(s: string | undefined) {
    if (s) {
        return true;
    }

    return false;
}

function truthyNullable(s: string | null) {
    return !!s;
}

function truthyAny(s: any) {
    return !!s;
}

function countWhile(s: string, steps: number) {
    let count = 0;
    while (s) {
        count++;
        s = count < steps ? s + "-" : "";
    }

    return count;
}

function main() {
    let e = "";
    let a = "a";

    assert(!truthy(e), "ternary on an empty string");
    assert(truthy(a), "ternary on a string");
    assert(!truthyIf(e), "if on an empty string");
    assert(truthyIf(a), "if on a string");
    assert(!e, "! of an empty string");
    assert(!!a, "!! of a string");
    if ("") {
        assert(false, "if on an empty literal");
    }

    assert((e || "x") == "x", "empty || x");
    assert((a || "x") == "a", "a || x");
    assert((e && "x") == "", "empty && x");
    assert((a && "x") == "x", "a && x");
    assert((e ?? "x") == "", "empty ?? x keeps the empty string");

    assert(countWhile("abc", 3) == 3, "while until the string is empty");
    assert(countWhile(e, 3) == 0, "while on an empty string");

    assert(!truthyOptional(e), "optional holding an empty string");
    assert(truthyOptional(a), "optional holding a string");
    assert(!truthyOptional(undefined), "undefined optional");

    assert(!truthyNullable(e), "nullable holding an empty string");
    assert(truthyNullable(a), "nullable holding a string");
    assert(!truthyNullable(null), "null");

    assert(!truthyAny(e), "any holding an empty string");
    assert(truthyAny(a), "any holding a string");

    assert(!globalEmpty, "global empty string");
    assert(globalPick == 2, "module-level ternary on an empty literal");

    assert(!(e + e), "empty concatenation");
    assert(!`${e}`, "empty template");

    print("done.");
}

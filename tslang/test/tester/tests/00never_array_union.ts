// `map` of a `never[] | string[]`: the union was read as a `never[]`, whose `map` took `item: never` ("'Never' can't
// be used as parameter type", TSLANG 193). Not under -mm=own, where a union of arrays mapped is refused ("a value that
// borrows an element cannot be stored").
function lengths(y: never[] | string[]) {
    let total = 0;
    for (const n of y.map(item => item.length)) {
        total += n;
    }

    return total;
}

// A function returning `never` given for one returning the type parameter `r`, inferred an `s32[]` from another argument
// once `never` gave way in the merge: "can't assign '(p0: string) => never' to '(p0: string) => s32[]'". It is called
// through an adapter, which gives nothing back as the function never returns.
function fail(message: string): never {
    throw message;
}

function withFew<a, r>(values: a[], haveFew: (values: a[]) => r, haveNone: (reason: string) => r): r {
    return values.length > 0 ? haveFew(values) : haveNone("No values.");
}

function id<a>(value: a): a {
    return value;
}

function main() {
    assert(lengths(["ab", "cde"]) == 5, "map of a never[] | string[]");
    assert(withFew([1, 2, 3], id, fail).length == 3, "a never-returning function given for an r-returning one");
    let caught = "";
    try {
        withFew([] as number[], id, fail);
    } catch (e) {
        caught = "" + e;
    }

    assert(caught == "No values.", "the never-returning function is called and throws");
    print("done.");
}

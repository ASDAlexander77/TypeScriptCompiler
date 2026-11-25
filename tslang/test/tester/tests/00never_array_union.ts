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

function main() {
    assert(lengths(["ab", "cde"]) == 5, "map of a never[] | string[]");
    print("done.");
}

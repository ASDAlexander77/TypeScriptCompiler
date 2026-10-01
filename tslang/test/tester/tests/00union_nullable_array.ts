// `number[] | null` has no tag: it is the array itself, null when it holds null. Reading the array
// out of it, and storing null in it, are casts the lowering has to do between the two
function lengthOf(p: number[] | null) {
    return p ? p.length : -1;
}

function first(p: number[] | null) {
    if (p !== null) {
        return p[0];
    }

    return -1;
}

function main() {
    let a: number[] = [1, 2];
    assert(lengthOf(a) == 2, "length of an array");
    assert(lengthOf(null) == -1, "length of null");
    assert(lengthOf([3, 4, 5]) == 3, "length of an array literal");
    assert(first(a) == 1, "first of an array");
    assert(first(null) == -1, "first of null");

    let b: number[] | null = null;
    assert(b === null, "null stored");
    b = a;
    assert(b !== null, "array stored");
    print("done.");
}

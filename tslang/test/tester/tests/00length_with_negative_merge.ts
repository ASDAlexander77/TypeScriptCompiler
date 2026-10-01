// A length (index, unsigned) and a negative integer in one inferred type: a return, a conditional
// and an array literal must all keep -1, not wrap it to 4294967295
function lengthFirst(p: string, found: boolean) {
    if (found) return p.length;
    return -1;
}

function negativeFirst(p: number[], found: boolean) {
    if (!found) return -1;
    return p.length;
}

function returns() {
    assert(lengthFirst("abc", true) == 3, "length first: length");
    assert(lengthFirst("abc", false) == -1, "length first: -1");
    assert(lengthFirst("abc", false) < 0, "length first: -1 < 0");
    assert(negativeFirst([1, 2], true) == 2, "-1 first: length");
    assert(negativeFirst([1, 2], false) == -1, "-1 first: -1");
}

function conditionals(p: string, found: boolean) {
    const a = found ? p.length : -1;
    const b = found ? -1 : p.length;
    assert(a == -1, "found ? length : -1");
    assert(b == 3, "found ? -1 : length");
}

function arrays(p: string) {
    const a = [p.length, -1];
    const b = [-1, p.length];
    assert(a[0] == 3 && a[1] == -1, "[length, -1]");
    assert(b[0] == -1 && b[1] == 3, "[-1, length]");
}

function constantString(found: boolean) {
    // a constant string's length folds to a constant
    const p = "abc";
    const a = found ? p.length : -1;
    assert(a == -1, "constant: found ? length : -1");
    assert([p.length, -1][1] == -1, "constant: [length, -1]");
}

function main() {
    returns();
    conditionals("abc", false);
    arrays("abc");
    constantString(false);
    print("done.");
}

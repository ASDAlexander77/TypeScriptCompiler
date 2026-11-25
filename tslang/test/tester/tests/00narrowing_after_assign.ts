// Narrowing and assignment (TSLANG 193):
// - a variable a test narrowed, then assigned in the branch, read the value the test saw (`x = "s" + x; print(x)`
//   printed the number), also after an `if` inside the branch that assigned it
// - after `if (!test) x = e;` the variable is what the test's being false leaves it or what the branch gave it:
//   `if (!(typeof d === "array")) d = [d];` makes it an array after the `if` ("access expression is not applicable")
class Item {
    constructor(public v: number) {}
}

function inBranch(x: number | string) {
    let seen = "";
    if (typeof x === "number") {
        x = "s" + x;
        seen = typeof x + ":" + x;
    }

    return seen + "/" + typeof x + ":" + x;
}

function nested(x: number | string, c: boolean) {
    let inner = "";
    if (typeof x === "number") {
        if (c) {
            x = "s" + x;
        }

        inner = typeof x + ":" + x;
    }

    return inner + "/" + typeof x + ":" + x;
}

function toArray(d: Item | Item[]) {
    if (!(typeof d === "array")) {
        d = [d];
    }

    let total = 0;
    for (const item of d) {
        total += item.v;
    }

    return total * 10 + d.length;
}

function dropNumber(x: number | string | boolean) {
    if (typeof x === "number") {
        x = "n" + x;
    }

    // a number no more: string | boolean
    if (typeof x === "boolean") {
        return -1;
    }

    return x.length;
}

function main() {
    assert(inBranch(5) == "string:s5/string:s5", "assigned in the branch");
    assert(inBranch("q") == "/string:q", "not assigned");
    assert(nested(1, true) == "string:s1/string:s1", "assigned in an if inside the branch");
    assert(nested(2, false) == "number:2/number:2", "not assigned in the if inside the branch");
    assert(toArray(new Item(4)) == 41 && toArray([new Item(1), new Item(2)]) == 32, "an array after the if");
    assert(dropNumber(12) == 3 && dropNumber("abcd") == 4 && dropNumber(true) == -1, "a number no more after the if");
    print("done.");
}

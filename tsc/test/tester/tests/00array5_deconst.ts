function arrayAssignment() {
    let [a, b, c] = [1, "foo", 3];
    assert(a == 1, "[]");
    assert(c == 3, "[]");
    assert(b == "foo", "[1]");
    [a, c] = [c, a];
    assert(a == 3, "[2]");
    assert(c == 1, "[]");

    const q = [4, 7];
    let p = 0;
    [a, c, p] = q;
    assert(a == 4, "[]");
    assert(c == 7, "[]");
    //assert(p === undefined, "[]");

    let [aa, [bb, cc]] = [4, [3, [1]]];
    assert(aa == 4, "[[]]");
    assert(bb == 3, "[[]]");
    assert(cc.length == 1, "[[]]");

    print("arrayAssignment done");
}

function main() {
    let [a, b] = [1, 2];

    print(a, b);

    [a, b] = [3, 4];

    print(a, b);

    arrayAssignment();

    print("done.");
}

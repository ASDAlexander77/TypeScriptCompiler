// `+` on strings copies each operand at its own offset with the length it measured, then ends the
// result with one terminator: empty and null operands, many operands, a result built in a loop,
// and results kept on the stack or the heap all hold exactly the bytes of their operands
function concat3(a: string, b: string, c: string) {
    return a + b + c;
}

function build(count: number) {
    let s = "";
    for (let i = 0; i < count; i++) {
        s = s + "ab" + "";
    }

    return s;
}

function main() {
    const empty = "";
    const a = "a";
    const bc = "bc";

    assert(concat3(a, bc, "def") == "abcdef", "three operands");
    assert(concat3(a, bc, "def").length == 6, "three operands: length");
    assert(concat3(empty, empty, empty) == "", "all empty");
    assert(concat3(empty, empty, empty).length == 0, "all empty: length");
    assert(concat3(empty, bc, empty) == "bc", "empty around a string");
    assert(concat3(a, empty, a) == "aa", "empty between strings");

    const many = a + bc + a + bc + a + bc + a + bc + a + bc;
    assert(many == "abcabcabcabcabc", "ten operands");
    assert(many.length == 15, "ten operands: length");

    const n: string | null = null;
    assert(a + n == "anull", "null operand prints as null");
    assert((n + empty).length == 4, "null then empty: length");

    const built = build(100);
    assert(built.length == 200, "built in a loop: length");
    assert(build(3) == "ababab", "built in a loop: content");

    const withNumber = a + 12 + bc;
    assert(withNumber == "a12bc", "number operand");

    print("done.");
}

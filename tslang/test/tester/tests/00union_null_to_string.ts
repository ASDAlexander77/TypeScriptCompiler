// `T | null` is T's own pointer, null when it holds null; turning it into a string has to look first
class C {
    toString() {
        return "C!";
    }
}

function nullString() {
    let y: string | null = null;
    assert(("a" + y) == "anull", "concat null string");
    assert(`${y}` == "null", "template null string");

    y = "b";
    assert(("a" + y) == "ab", "concat string");
    assert(`${y}` == "b", "template string");
}

function isNull(p: string) {
    return p == null;
}

// only printing and concatenating read null as "null": converted to a string, it stays null
function nullStringStaysNull() {
    let y: string | null = null;
    let s: string = y;
    assert(s == null, "assigned null string");
    assert(isNull(y), "null string argument");
}

function nullClass() {
    let c: C | null = null;
    assert(("a" + c) == "anull", "concat null class");

    c = new C();
    assert(("a" + c) == "aC!", "concat class");
    assert(`${c}` == "C!", "template class");
}

function main() {
    let y: string | null = null;
    print(y);

    let c: C | null = null;
    print(c);
    c = new C();
    print(c);

    nullString();
    nullStringStaysNull();
    nullClass();

    print("done.");
}

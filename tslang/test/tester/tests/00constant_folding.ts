// Expressions of constants are worked out by the compiler: `"a" + "b"`, `1 + "a"`, `${1 + 1}` are
// string constants, built at no run-time cost, and `1 + 2 + "x"` folds as well. Each is checked
// against the same expression of values the compiler cannot see (parameters), so the folded text
// is the run-time conversion's.

function text(v: number): string {
    return "" + v;
}

function join(a: string, b: string): string {
    return a + b;
}

function addInt(a: int, b: int): int {
    return a + b;
}

class Holder {
    name: string;
    constructor(name: string) {
        this.name = name;
    }
}

function makeName(): string {
    return "na" + "me";
}

function restText(...values: number[]): string {
    let text = "" + values.length + ":";
    for (const v of values) {
        text = text + " " + v;
    }
    return text;
}

function restStrings(...values: string[]): string {
    let text = "";
    for (const v of values) {
        text = text + v + ";";
    }
    return text;
}

function main() {
    // strings and strings
    assert("a" + "b" == "ab", "string + string");
    assert("a" + "b" + "c" == join(join("a", "b"), "c"), "three strings");
    const greeting = "hi";
    assert(greeting + " there" == "hi there", "const string + string");

    // numbers and strings, as JS writes them
    assert(1 + "a" == "1a", "int + string");
    assert("x" + 1 + 2 == "x12", "string + int + int");
    assert(1 + 2 + "x" == "3x", "int + int + string");
    assert("v=" + true == "v=true", "string + boolean");
    assert("v=" + false == "v=false", "string + false");

    assert("" + 0.1 == text(0.1), "0.1");
    assert("" + 1.5 == text(1.5), "1.5");
    assert("" + 100.25 == text(100.25), "100.25");
    assert("" + 1e21 == text(1e21), "1e21");
    assert("" + 1e20 == text(1e20), "1e20");
    assert("" + 1e-7 == text(1e-7), "1e-7");
    assert("" + 0.000001 == text(0.000001), "1e-6");
    assert("" + 123e-20 == text(123e-20), "123e-20");
    assert("" + 1.7976931348623157e308 == text(1.7976931348623157e308), "max");
    assert("" + 5e-324 == text(5e-324), "min");

    // number arithmetic folds in IEEE double, as at run time
    const tenth = 0.1;
    const fifth = 0.2;
    assert("" + (0.1 + 0.2) == text(tenth + fifth), "0.1 + 0.2");
    assert(0.1 + 0.2 == tenth + fifth, "0.1 + 0.2 number");
    assert("" + (1.0 / 3.0) == text(1.0 / 3.0), "1/3");
    assert("" + (1e300 * 1e300) == "Infinity", "overflow");
    assert("" + (2.5 * 4.0) == "10", "2.5 * 4");

    // integers wrap at their width, as the run-time op does
    assert(1 + 2 == 3, "1 + 2");
    assert((1 << 4) | 3, "bits");
    assert(((1 << 4) | 3) == 19, "bits value");
    assert(2147483647 + 1 == addInt(2147483647, 1), "wrap");
    assert(6 * 7 - 2 == 40, "6 * 7 - 2");
    assert((-16 >> 2) == -4, "arithmetic shift");

    // templates
    assert(`n=${1 + 1}` == "n=2", "template of int");
    assert(`${"a"}${"b"}` == "ab", "template of strings");
    assert(`${true}/${1.5}` == "true/1.5", "template of boolean and number");
    assert(`x${0.1 + 0.2}` == "x" + text(tenth + fifth), "template of sum");

    // the folded values have the operation's type, not a literal type
    let s = "a" + "b";
    s = "something else";
    assert(s == "something else", "string variable");
    let n = 1 + 2;
    n = 7;
    assert(n == 7, "int variable");
    let f = 0.5 + 0.25;
    f = 2.5;
    assert(f == 2.5, "number variable");

    // a folded string goes where a built one does
    const holder = new Holder("ho" + "lder");
    assert(holder.name == "holder", "field");
    const names: string[] = [];
    names.push("a" + 1);
    names.push(makeName());
    assert(names[0] == "a1" && names[1] == "name", "array");
    let built = "a" + "b";
    for (let i = 0; i < 3; i++) {
        built = built + i;
    }
    assert(built == "ab012", "grown from a folded string");

    // folded constants as rest arguments: every argument a constant of the element's type made a
    // static array literal, which the call passed as one element of garbage
    const scale = 1.5;
    const three = 2.0 * scale;
    assert(restText(three, three) == "2: 3 3", "the same folded number twice");
    assert(restText(three, 0.5 * scale) == "2: 3 0.75", "two folded numbers");
    assert(restText(three) == "1: 3", "one folded number");
    assert(restText(three, 2.0 * scale) == "2: 3 3", "the same value folded twice");
    const ab = "a" + "b";
    assert(restStrings(ab, ab, "c" + "d") == "ab;ab;cd;", "folded strings");

    print("done.");
}

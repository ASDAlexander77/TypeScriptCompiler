// What a string or a number may still be given, now that a string given to a number (or back) is an
// error (assign_*_error.ts, argument_*_error.ts): what TypeScript allows.

enum E {
    A,
    B
}

// an untyped parameter is `any` in TypeScript: given a number, it may still go where a string is
// wanted (tslang makes the function generic in it, and instantiates it for a number)
function circumference(r) {
    return parseFloat(r) * 2;
}

// a string or a number goes to a parameter of a union of both
function show(v: number | string) {
    return "" + v;
}

function main() {
    assert(circumference(1.5) == 3 && circumference("2.5") == 5, "an untyped parameter");
    assert(show(1) == "1" && show("a") == "a", "a union parameter");

    // a variable a test narrowed takes anything its declaration allows
    let u: number | string = 1;
    if (typeof u === "number") {
        u = "now a string";
    }
    assert(u == "now a string", "a narrowed variable");

    // a string plus a number is a string
    let s = "a";
    s += 1;
    s = s + 2;
    assert(s == "a12", "a string plus a number");

    // `any` is converted when it is read
    const w: any = 3;
    let n: number = w;
    n = w;
    assert(n == 3, "any into a number");

    // an enum member is a number
    let e: number = E.B;
    assert(e == 1, "an enum into a number");
    e = E.A;

    // a character is a string
    const str = "xyz";
    let c: string = str[0];
    c = str[1];
    assert(c == "y", "a character into a string");

    // numbers of any width
    let i: int = 5;
    i = 7;
    let f: number = i;
    f = 2.5;
    assert(i == 7 && f == 2.5, "numbers of other widths");

    print("done.");
}
